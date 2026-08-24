"""
Multi-Object Tracking Module for Football Analysis (Milestone 2 & Kalman Enhancements).
Uses ByteTrack with Kalman Filter motion prediction, center extrapolation coasting,
and automated memory reclamation.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import supervision as sv
from supervision import ByteTrack, Detections

from ..detection.detector import DetectionResult

# Backwards compatibility alias
TrackResult = DetectionResult


class PlayerTracker:
    """
    ByteTrack-based Multi-Object Tracker with Kalman Filter motion prediction,
    lost-track coasting (interpolation across dropouts), and historical movement trails.
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.tracker = ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=frame_rate,
            )
        self.trail_length = trail_length
        self.enable_coasting = enable_coasting
        self.max_coast_frames = max_coast_frames

        # Trail storage: {track_id: deque([(x_center, y_bottom), ...])}
        self.trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.trail_length))

        # Last known positions and lost frame counters for coasting
        self.last_boxes: Dict[int, np.ndarray] = {}
        self.last_velocities: Dict[int, np.ndarray] = {}
        self.lost_counters: Dict[int, int] = {}

    def update(self, detection_result: DetectionResult) -> DetectionResult:
        """
        Update tracker with detections and emit persistent track IDs with Kalman coasting.
        """
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
            tracked_tids = tracked_sv.tracker_id if tracked_sv.tracker_id is not None else np.full(len(tracked_sv.xyxy), -1, dtype=int)
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

        # Kalman Coasting across temporary single-frame dropouts
        if self.enable_coasting:
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
