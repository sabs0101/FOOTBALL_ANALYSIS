"""
Multi-Object Tracking Module for Football Analysis (Milestones 2, 9 CMC & 10 Cut Re-ID).
Uses ByteTrack with Kalman Filter motion prediction, Camera Motion Compensation (CMC),
Camera Cut awareness, long-term Re-ID identity preservation, and lost-track coasting.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Union
import warnings
import cv2
import numpy as np
import supervision as sv
from supervision import ByteTrack, Detections

from ..detection.detector import DetectionResult
from .reid import PlayerReID, ReIDMatchResult

# Backwards compatibility alias
TrackResult = DetectionResult


class PlayerTracker:
    """
    ByteTrack-based Multi-Object Tracker with Kalman Filter motion prediction,
    Camera Motion Compensation (CMC), Camera Cut Boundary resets, Re-ID cross-cut
    re-association, and historical movement trails.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.18,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.70,
        frame_rate: int = 25,
        trail_length: int = 20,
        enable_coasting: bool = True,
        max_coast_frames: int = 4,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.frame_rate = frame_rate
        self.trail_length = trail_length
        self.enable_coasting = enable_coasting
        self.max_coast_frames = max_coast_frames

        self._init_tracker()

        # Trail storage: {track_id: deque([(x_center, y_bottom), ...])}
        self.trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.trail_length))

        # Last known positions and lost frame counters for coasting
        self.last_boxes: Dict[int, np.ndarray] = {}
        self.last_velocities: Dict[int, np.ndarray] = {}
        self.lost_counters: Dict[int, int] = {}

    def _init_tracker(self):
        """Initialize the underlying ByteTrack instance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.tracker = ByteTrack(
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
                frame_rate=self.frame_rate,
            )

    def reset_tracks(self):
        """
        Reset tracker state and motion buffers across camera cuts or scene transitions.
        Prevents Kalman state extrapolation across disconnected scenes.
        """
        self._init_tracker()
        self.last_boxes.clear()
        self.last_velocities.clear()
        self.lost_counters.clear()
        self.trails.clear()

    def _apply_camera_motion_compensation(self, camera_transform: np.ndarray):
        """
        Warp cached previous boxes and velocities by the affine camera transformation
        matrix M_{t-1 -> t} to align them with the current camera coordinate frame.
        """
        if camera_transform is None or not np.isfinite(camera_transform).all():
            return

        M = camera_transform[:2, :]

        # Warp last known bounding boxes
        for tid, box in list(self.last_boxes.items()):
            corners = np.array([
                [box[0], box[1]],
                [box[2], box[1]],
                [box[2], box[3]],
                [box[0], box[3]],
            ], dtype=np.float32).reshape(-1, 1, 2)

            warped = cv2.transform(corners, M).reshape(-1, 2)
            self.last_boxes[tid] = np.array([
                np.min(warped[:, 0]),
                np.min(warped[:, 1]),
                np.max(warped[:, 0]),
                np.max(warped[:, 1]),
            ], dtype=np.float32)

    def update(
        self,
        detection_result: DetectionResult,
        camera_transform: Optional[np.ndarray] = None,
        is_cut: bool = False,
        reid: Optional[PlayerReID] = None,
        frame: Optional[np.ndarray] = None,
        candidate_teams: Optional[List[int]] = None,
        candidate_roles: Optional[List[int]] = None,
        candidate_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
    ) -> DetectionResult:
        """
        Update tracker with detections, perform Camera Motion Compensation (CMC),
        Camera Cut resets, and Re-ID identity preservation across scene transitions.
        """
        # If a camera cut occurred, reset motion state
        if is_cut:
            self.reset_tracks()

        # Apply Camera Motion Compensation (only when no cut occurred)
        if camera_transform is not None and not is_cut:
            self._apply_camera_motion_compensation(camera_transform)

        if len(detection_result.xyxy) == 0:
            return detection_result

        player_mask = detection_result.class_ids == 0
        ball_mask = detection_result.class_ids == 32

        player_boxes = detection_result.xyxy[player_mask]
        player_confs = detection_result.confidences[player_mask]
        player_cids = detection_result.class_ids[player_mask]

        if len(player_boxes) > 0:
            sv_detections = Detections(
                xyxy=player_boxes,
                confidence=player_confs,
                class_id=player_cids,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                tracked_sv = self.tracker.update_with_detections(sv_detections)

            tracked_boxes = tracked_sv.xyxy
            tracked_confs = tracked_sv.confidence if tracked_sv.confidence is not None else player_confs[:len(tracked_sv.xyxy)]
            tracked_cids = tracked_sv.class_id if tracked_sv.class_id is not None else np.zeros(len(tracked_sv.xyxy), dtype=int)
            tracked_tids = tracked_sv.tracker_id.copy() if tracked_sv.tracker_id is not None else np.full(len(tracked_sv.xyxy), -1, dtype=int)

            # Cross-Cut Re-ID Matching: If a cut occurred, re-assign gallery IDs
            if is_cut and reid is not None and frame is not None and len(tracked_boxes) > 0:
                reid_res = reid.reassociate(
                    frame=frame,
                    boxes=tracked_boxes,
                    candidate_teams=candidate_teams,
                    candidate_roles=candidate_roles,
                    candidate_positions=candidate_positions,
                    frame_idx=detection_result.frame_idx,
                )
                for det_idx, gallery_tid in reid_res.matches.items():
                    if det_idx < len(tracked_tids):
                        tracked_tids[det_idx] = gallery_tid

            # Update Re-ID gallery with current active tracks
            if reid is not None and frame is not None and len(tracked_boxes) > 0:
                reid.update_gallery(
                    frame=frame,
                    boxes=tracked_boxes,
                    track_ids=tracked_tids,
                    team_ids=np.array(candidate_teams) if candidate_teams is not None else None,
                    role_ids=np.array(candidate_roles) if candidate_roles is not None else None,
                    pitch_positions=candidate_positions,
                    frame_idx=detection_result.frame_idx,
                )
        else:
            tracked_boxes = np.empty((0, 4), dtype=np.float32)
            tracked_confs = np.empty((0,), dtype=np.float32)
            tracked_cids = np.empty((0,), dtype=int)
            tracked_tids = np.empty((0,), dtype=int)

        # Update trails & velocity estimates
        active_tids = set()
        for i, tid in enumerate(tracked_tids):
            if tid >= 0:
                active_tids.add(tid)
                box = tracked_boxes[i]
                foot_x = (box[0] + box[2]) / 2.0
                foot_y = float(box[3])
                self.trails[tid].append((foot_x, foot_y))

                if tid in self.last_boxes:
                    prev_center = (self.last_boxes[tid][:2] + self.last_boxes[tid][2:]) / 2.0
                    curr_center = (box[:2] + box[2:]) / 2.0
                    vel = curr_center - prev_center
                    self.last_velocities[tid] = vel

                self.last_boxes[tid] = box.copy()
                self.lost_counters[tid] = 0

        # Kalman Coasting across temporary single-frame dropouts (only in non-cut frames)
        if self.enable_coasting and not is_cut:
            coasted_boxes = []
            coasted_confs = []
            coasted_cids = []
            coasted_tids = []

            for tid, last_box in list(self.last_boxes.items()):
                if tid not in active_tids:
                    count = self.lost_counters.get(tid, 0) + 1
                    self.lost_counters[tid] = count

                    if count <= self.max_coast_frames:
                        vel = self.last_velocities.get(tid, np.zeros(2, dtype=np.float32))
                        damped_vel = vel * (0.85 ** count)

                        # Lock bounding box dimensions (w, h) and update center
                        bw = last_box[2] - last_box[0]
                        bh = last_box[3] - last_box[1]
                        cx = (last_box[0] + last_box[2]) / 2.0 + damped_vel[0]
                        cy = (last_box[1] + last_box[3]) / 2.0 + damped_vel[1]

                        new_box = np.array([
                            cx - bw / 2.0,
                            cy - bh / 2.0,
                            cx + bw / 2.0,
                            cy + bh / 2.0,
                        ], dtype=np.float32)

                        self.last_boxes[tid] = new_box
                        coasted_boxes.append(new_box)
                        coasted_confs.append(0.50)
                        coasted_cids.append(0)
                        coasted_tids.append(tid)

                        foot_x = cx
                        foot_y = new_box[3]
                        self.trails[tid].append((foot_x, foot_y))
                    else:
                        # Automated memory cleanup after 45 inactive frames
                        if count > 45:
                            self.last_boxes.pop(tid, None)
                            self.last_velocities.pop(tid, None)
                            self.lost_counters.pop(tid, None)
                            self.trails.pop(tid, None)

            if coasted_boxes:
                if len(tracked_boxes) > 0:
                    tracked_boxes = np.vstack([tracked_boxes, np.array(coasted_boxes, dtype=np.float32)])
                    tracked_confs = np.concatenate([tracked_confs, np.array(coasted_confs, dtype=np.float32)])
                    tracked_cids = np.concatenate([tracked_cids, np.array(coasted_cids, dtype=int)])
                    tracked_tids = np.concatenate([tracked_tids, np.array(coasted_tids, dtype=int)])
                else:
                    tracked_boxes = np.array(coasted_boxes, dtype=np.float32)
                    tracked_confs = np.array(coasted_confs, dtype=np.float32)
                    tracked_cids = np.array(coasted_cids, dtype=int)
                    tracked_tids = np.array(coasted_tids, dtype=int)

        # Merge back Ball detections
        ball_boxes = detection_result.xyxy[ball_mask]
        ball_confs = detection_result.confidences[ball_mask]
        ball_cids = detection_result.class_ids[ball_mask]
        ball_tids = np.full(len(ball_boxes), -1, dtype=int)

        if len(ball_boxes) > 0:
            if len(tracked_boxes) > 0:
                final_boxes = np.vstack([tracked_boxes, ball_boxes])
                final_confs = np.concatenate([tracked_confs, ball_confs])
                final_cids = np.concatenate([tracked_cids, ball_cids])
                final_tids = np.concatenate([tracked_tids, ball_tids])
            else:
                final_boxes = ball_boxes
                final_confs = ball_confs
                final_cids = ball_cids
                final_tids = ball_tids
        else:
            final_boxes = tracked_boxes
            final_confs = tracked_confs
            final_cids = tracked_cids
            final_tids = tracked_tids

        cnames = ["person" if cid == 0 else "sports_ball" for cid in final_cids]

        return DetectionResult(
            xyxy=final_boxes,
            confidences=final_confs,
            class_ids=final_cids,
            class_names=cnames,
            tracker_ids=final_tids,
            trails=dict(self.trails),
            frame_idx=detection_result.frame_idx,
        )
