"""
Ball Tracking, Trajectory Smoothing, Interpolation, and Player Possession Assignment Module.
Implements Kalman Filter state estimation, physical outlier rejection, short-gap trajectory
interpolation across detector dropouts, and metric-space player possession assignment with
temporal debouncing (hysteresis) and turnover tracking.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from ..detection.detector import DetectionResult


@dataclass
class BallState:
    """Represents the instantaneous tracked state of the football."""
    frame_idx: int
    box_xyxy: Optional[np.ndarray] = None
    center_px: Optional[Tuple[float, float]] = None
    position_m: Optional[Tuple[float, float]] = None
    speed_kmh: float = 0.0
    is_interpolated: bool = False
    confidence: float = 0.0


@dataclass
class PossessionResult:
    """Represents tactical ball possession telemetry for a given frame and cumulative match."""
    frame_idx: int
    possessing_player_id: Optional[int] = None
    possessing_team_id: Optional[int] = None  # 0: Team A, 1: Team B, None: Loose/Contested
    possessing_team_name: Optional[str] = None
    distance_to_ball_m: float = 999.0
    is_contested: bool = False
    team_a_possession_pct: float = 50.0
    team_b_possession_pct: float = 50.0
    contested_pct: float = 0.0
    turnover_count: int = 0
    player_possession_counts: Dict[int, int] = field(default_factory=dict)


class BallKalmanFilter:
    """
    2D Constant-Velocity Kalman Filter for smoothing and projecting ball coordinates.
    State: [x, y, vx, vy]^T
    """

    def __init__(self, dt: float = 1.0 / 25.0, process_noise_std: float = 8.0, meas_noise_std: float = 0.8):
        self.dt = dt
        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Measurement matrix H (we observe x, y)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # Process noise covariance Q
        q = process_noise_std ** 2
        dt4 = (dt ** 4) / 4.0 * q
        dt3 = (dt ** 3) / 2.0 * q
        dt2 = (dt ** 2) * q
        self.Q = np.array([
            [dt4, 0.0, dt3, 0.0],
            [0.0, dt4, 0.0, dt3],
            [dt3, 0.0, dt2, 0.0],
            [0.0, dt3, 0.0, dt2],
        ], dtype=np.float32)

        # Measurement noise covariance R
        r = meas_noise_std ** 2
        self.R = np.eye(2, dtype=np.float32) * r

        # State vector and error covariance
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.diag([10.0, 10.0, 1000.0, 1000.0]).astype(np.float32)
        self.is_initialized = False

    def initialize(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """Initialize filter with initial coordinate measurement."""
        self.x = np.array([[x], [y], [vx], [vy]], dtype=np.float32)
        self.P = np.diag([10.0, 10.0, 1000.0, 1000.0]).astype(np.float32)
        self.is_initialized = True

    def predict(self) -> Tuple[float, float]:
        """Project state forward by one time step."""
        if not self.is_initialized:
            return (0.0, 0.0)
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Incorporate measurement and update state estimate."""
        if not self.is_initialized:
            self.initialize(x, y)
            return (x, y)

        z = np.array([[x], [y]], dtype=np.float32)
        y_residual = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y_residual)
        I = np.eye(4, dtype=np.float32)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    @property
    def velocity(self) -> Tuple[float, float]:
        """Return estimated velocity (vx, vy)."""
        return (float(self.x[2, 0]), float(self.x[3, 0]))


class BallTracker:
    """
    Robust Ball Tracking, Gap Interpolation, Trajectory Smoothing,
    and Player Possession Assignment Engine.
    """

    def __init__(
        self,
        fps: float = 25.0,
        max_interpolation_gap: int = 6,
        max_speed_kmh: float = 140.0,
        possession_radius_m: float = 1.8,
        trail_length: int = 25,
        hysteresis_frames: int = 2,
    ):
        self.fps = max(1.0, float(fps))
        self.dt = 1.0 / self.fps
        self.max_interpolation_gap = max_interpolation_gap
        self.max_speed_kmh = max_speed_kmh
        self.max_speed_mps = (max_speed_kmh * 1000.0) / 3600.0
        self.possession_radius_m = possession_radius_m
        self.trail_length = trail_length
        self.hysteresis_frames = hysteresis_frames

        # Kalman filters for Pixel and Metric tracking
        self.kf_pixel = BallKalmanFilter(dt=self.dt, process_noise_std=8.0, meas_noise_std=0.8)
        self.kf_metric = BallKalmanFilter(dt=self.dt, process_noise_std=3.0, meas_noise_std=0.4)

        # State history
        self.last_valid_state: Optional[BallState] = None
        self.consecutive_lost_frames: int = 0
        self.last_detection_frame: int = -1

        # Trajectory trail deques
        # Stores: (x_px, y_px, x_m, y_m, speed_kmh, is_interpolated)
        self.trail: Deque[Tuple[float, float, Optional[float], Optional[float], float, bool]] = deque(maxlen=self.trail_length)

        # Possession state & debouncing
        self.current_possessor_id: Optional[int] = None
        self.current_possessor_team: Optional[int] = None
        self.candidate_possessor_id: Optional[int] = None
        self.candidate_frames: int = 0
        self.turnover_count: int = 0

        # Match cumulative possession stats
        self.team_a_possession_frames: int = 0
        self.team_b_possession_frames: int = 0
        self.contested_frames: int = 0
        self.total_tracked_frames: int = 0
        self.player_possession_counts: Dict[int, int] = {}

    def is_outlier(
        self,
        new_px: Tuple[float, float],
        new_m: Optional[Tuple[float, float]],
        frames_elapsed: int = 1,
    ) -> bool:
        """
        Determine if the new ball detection is physically implausible (teleportation/misdetection).
        """
        if self.last_valid_state is None:
            return False

        dt = max(1e-3, frames_elapsed * self.dt)

        # 1. Metric space speed check
        if new_m is not None and self.last_valid_state.position_m is not None:
            dx = new_m[0] - self.last_valid_state.position_m[0]
            dy = new_m[1] - self.last_valid_state.position_m[1]
            dist_m = float(np.hypot(dx, dy))
            speed_mps = dist_m / dt
            if speed_mps > self.max_speed_mps * 1.15:  # Tolerance buffer
                return True

        # 2. Pixel space displacement check (extreme jump > 380px in 1 frame)
        if self.last_valid_state.center_px is not None:
            dpx = new_px[0] - self.last_valid_state.center_px[0]
            dpy = new_px[1] - self.last_valid_state.center_px[1]
            dist_px = float(np.hypot(dpx, dpy))
            max_allowed_px = 380.0 * max(1, frames_elapsed)
            if dist_px > max_allowed_px:
                return True

        return False

    def update(
        self,
        detections: DetectionResult,
        homography_matrix: Optional[np.ndarray] = None,
        player_positions_m: Optional[np.ndarray] = None,
        player_track_ids: Optional[np.ndarray] = None,
        player_team_ids: Optional[np.ndarray] = None,
        frame_idx: int = 0,
    ) -> Tuple[Optional[BallState], PossessionResult]:
        """
        Process the current frame detections, perform Kalman state estimation,
        smooth ball trajectory, interpolate gaps, and assign player possession.
        """
        self.total_tracked_frames += 1
        ball_res = detections.get_ball()
        has_detection = len(ball_res.xyxy) > 0

        current_ball_state: Optional[BallState] = None

        if has_detection:
            # 1. Raw Detection Processing
            raw_box = ball_res.xyxy[0]
            raw_conf = float(ball_res.confidences[0]) if len(ball_res.confidences) > 0 else 0.5
            raw_cx = float((raw_box[0] + raw_box[2]) / 2.0)
            raw_cy = float((raw_box[1] + raw_box[3]) / 2.0)
            raw_px = (raw_cx, raw_cy)

            # Project to metric coordinates if homography is available
            raw_m: Optional[Tuple[float, float]] = None
            if homography_matrix is not None:
                pt_img = np.array([[[raw_cx, raw_cy]]], dtype=np.float32)
                pt_m = cv2.perspectiveTransform(pt_img, homography_matrix)
                if pt_m is not None and len(pt_m) > 0:
                    raw_m = (float(pt_m[0, 0, 0]), float(pt_m[0, 0, 1]))

            # Check outlier
            frames_since = (frame_idx - self.last_detection_frame) if self.last_detection_frame >= 0 else 1
            if self.is_outlier(raw_px, raw_m, frames_elapsed=frames_since):
                # Reject outlier as false positive, treat as missing detection
                has_detection = False
            else:
                # Valid detection: Kalman Update
                if not self.kf_pixel.is_initialized:
                    self.kf_pixel.initialize(raw_cx, raw_cy)
                    smooth_cx, smooth_cy = raw_cx, raw_cy
                else:
                    self.kf_pixel.predict()
                    smooth_cx, smooth_cy = self.kf_pixel.update(raw_cx, raw_cy)

                smooth_m: Optional[Tuple[float, float]] = None
                speed_kmh = 0.0

                if raw_m is not None:
                    if not self.kf_metric.is_initialized:
                        self.kf_metric.initialize(raw_m[0], raw_m[1])
                        smooth_m = raw_m
                    else:
                        self.kf_metric.predict()
                        smooth_mx, smooth_my = self.kf_metric.update(raw_m[0], raw_m[1])
                        smooth_m = (smooth_mx, smooth_my)

                    vx, vy = self.kf_metric.velocity
                    speed_mps = float(np.hypot(vx, vy))
                    speed_kmh = round(speed_mps * 3.6, 1)

                bw = raw_box[2] - raw_box[0]
                bh = raw_box[3] - raw_box[1]
                smooth_box = np.array([
                    smooth_cx - bw / 2.0,
                    smooth_cy - bh / 2.0,
                    smooth_cx + bw / 2.0,
                    smooth_cy + bh / 2.0,
                ], dtype=np.float32)

                current_ball_state = BallState(
                    frame_idx=frame_idx,
                    box_xyxy=smooth_box,
                    center_px=(smooth_cx, smooth_cy),
                    position_m=smooth_m,
                    speed_kmh=speed_kmh,
                    is_interpolated=False,
                    confidence=raw_conf,
                )

                self.last_valid_state = current_ball_state
                self.consecutive_lost_frames = 0
                self.last_detection_frame = frame_idx
                self.trail.append((
                    smooth_cx,
                    smooth_cy,
                    smooth_m[0] if smooth_m else None,
                    smooth_m[1] if smooth_m else None,
                    speed_kmh,
                    False,
                ))

        if not has_detection:
            # 2. Kalman Prediction & Gap Interpolation
            self.consecutive_lost_frames += 1

            if (
                self.last_valid_state is not None
                and self.consecutive_lost_frames <= self.max_interpolation_gap
                and self.kf_pixel.is_initialized
            ):
                pred_cx, pred_cy = self.kf_pixel.predict()

                pred_m: Optional[Tuple[float, float]] = None
                speed_kmh = 0.0
                if self.kf_metric.is_initialized:
                    pred_mx, pred_my = self.kf_metric.predict()
                    pred_m = (pred_mx, pred_my)
                    vx, vy = self.kf_metric.velocity
                    speed_mps = float(np.hypot(vx, vy))
                    speed_kmh = round(speed_mps * 3.6, 1)

                # Maintain last bounding box size
                last_box = self.last_valid_state.box_xyxy
                bw = (last_box[2] - last_box[0]) if last_box is not None else 18.0
                bh = (last_box[3] - last_box[1]) if last_box is not None else 18.0

                interp_box = np.array([
                    pred_cx - bw / 2.0,
                    pred_cy - bh / 2.0,
                    pred_cx + bw / 2.0,
                    pred_cy + bh / 2.0,
                ], dtype=np.float32)

                current_ball_state = BallState(
                    frame_idx=frame_idx,
                    box_xyxy=interp_box,
                    center_px=(pred_cx, pred_cy),
                    position_m=pred_m,
                    speed_kmh=speed_kmh,
                    is_interpolated=True,
                    confidence=max(0.20, self.last_valid_state.confidence * (0.85 ** self.consecutive_lost_frames)),
                )

                self.last_valid_state = current_ball_state
                self.trail.append((
                    pred_cx,
                    pred_cy,
                    pred_m[0] if pred_m else None,
                    pred_m[1] if pred_m else None,
                    speed_kmh,
                    True,
                ))
            else:
                # Beyond interpolation window -> Ball lost
                current_ball_state = None

        # 3. Player Possession Assignment
        possession_result = self._assign_possession(
            current_ball_state=current_ball_state,
            player_positions_m=player_positions_m,
            player_track_ids=player_track_ids,
            player_team_ids=player_team_ids,
            frame_idx=frame_idx,
        )

        return current_ball_state, possession_result

    def _assign_possession(
        self,
        current_ball_state: Optional[BallState],
        player_positions_m: Optional[np.ndarray],
        player_track_ids: Optional[np.ndarray],
        player_team_ids: Optional[np.ndarray],
        frame_idx: int,
    ) -> PossessionResult:
        """
        Assign on-ball player possession using metric proximity, contest detection,
        and temporal hysteresis.
        """
        min_dist = 999.0
        candidate_player_id: Optional[int] = None
        candidate_team_id: Optional[int] = None
        is_contested = False

        if (
            current_ball_state is not None
            and current_ball_state.position_m is not None
            and player_positions_m is not None
            and len(player_positions_m) > 0
            and player_track_ids is not None
            and player_team_ids is not None
        ):
            bx, by = current_ball_state.position_m
            # Euclidean distances to all players on pitch
            dists = np.hypot(player_positions_m[:, 0] - bx, player_positions_m[:, 1] - by)
            sorted_indices = np.argsort(dists)

            closest_idx = sorted_indices[0]
            min_dist = float(dists[closest_idx])

            if min_dist <= self.possession_radius_m:
                candidate_player_id = int(player_track_ids[closest_idx])
                candidate_team_id = int(player_team_ids[closest_idx])

                # Check contest: second closest player is from opposing team and within contested threshold
                if len(sorted_indices) > 1:
                    second_idx = sorted_indices[1]
                    second_dist = float(dists[second_idx])
                    second_team = int(player_team_ids[second_idx])
                    if second_team != candidate_team_id and (second_dist <= self.possession_radius_m + 0.40):
                        is_contested = True

        # Hysteresis & Debouncing logic
        if candidate_player_id is not None:
            if candidate_player_id == self.current_possessor_id:
                # Same player holding possession
                self.candidate_frames = 0
            else:
                # New candidate player
                if candidate_player_id == self.candidate_possessor_id:
                    self.candidate_frames += 1
                else:
                    self.candidate_possessor_id = candidate_player_id
                    self.candidate_frames = 1

                # Turnover transition trigger
                immediate_takeover = min_dist < 1.05  # Direct close tackle/dribble
                held_candidate = self.candidate_frames >= self.hysteresis_frames

                if immediate_takeover or held_candidate:
                    if (
                        self.current_possessor_team is not None
                        and candidate_team_id is not None
                        and candidate_team_id != self.current_possessor_team
                    ):
                        self.turnover_count += 1

                    self.current_possessor_id = candidate_player_id
                    self.current_possessor_team = candidate_team_id
                    self.candidate_frames = 0
        else:
            # Ball is loose / free
            self.candidate_possessor_id = None
            self.candidate_frames = 0
            if min_dist > self.possession_radius_m * 1.5:
                # Disengage possession once ball moves distinctly clear
                self.current_possessor_id = None
                self.current_possessor_team = None

        # Cumulative match accounting
        if is_contested:
            self.contested_frames += 1
        elif self.current_possessor_team == 0:
            self.team_a_possession_frames += 1
        elif self.current_possessor_team == 1:
            self.team_b_possession_frames += 1

        if self.current_possessor_id is not None:
            self.player_possession_counts[self.current_possessor_id] = (
                self.player_possession_counts.get(self.current_possessor_id, 0) + 1
            )

        # Compute match percentages
        active_possession_total = self.team_a_possession_frames + self.team_b_possession_frames
        if active_possession_total > 0:
            pct_a = round((self.team_a_possession_frames / active_possession_total) * 100.0, 1)
            pct_b = round((self.team_b_possession_frames / active_possession_total) * 100.0, 1)
        else:
            pct_a, pct_b = 50.0, 50.0

        contested_pct = round(
            (self.contested_frames / max(1, self.total_tracked_frames)) * 100.0, 1
        )

        team_name: Optional[str] = None
        if is_contested:
            team_name = "Contested 50-50"
        elif self.current_possessor_team == 0:
            team_name = "Team A"
        elif self.current_possessor_team == 1:
            team_name = "Team B"

        return PossessionResult(
            frame_idx=frame_idx,
            possessing_player_id=self.current_possessor_id,
            possessing_team_id=self.current_possessor_team,
            possessing_team_name=team_name,
            distance_to_ball_m=round(min_dist, 2),
            is_contested=is_contested,
            team_a_possession_pct=pct_a,
            team_b_possession_pct=pct_b,
            contested_pct=contested_pct,
            turnover_count=self.turnover_count,
            player_possession_counts=dict(self.player_possession_counts),
        )
