"""
Physical Kinematics, Velocity Estimation, and Distance Calculation (Polished & Hardened).
Uses Least-Squares Linear Regression Velocity Estimation with Outlier Residual Rejection
across a 0.72s sliding window, FIFA speed category classification, and noise deadband filtering.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class PlayerMetrics:
    """
    Kinematic metrics for a single tracked player.

    Attributes:
        track_id: Persistent tracking ID.
        current_speed_kmh: Instantaneous regression velocity in km/h.
        smoothed_speed_kmh: EMA smoothed speed in km/h.
        cumulative_distance_m: Total physical distance traversed in meters.
        top_speed_kmh: Peak sprint speed recorded in km/h.
        activity_category: FIFA category ("Standing", "Walking", "Jogging", "Running", "Sprinting").
    """
    track_id: int
    current_speed_kmh: float = 0.0
    smoothed_speed_kmh: float = 0.0
    cumulative_distance_m: float = 0.0
    top_speed_kmh: float = 0.0
    activity_category: str = "Standing"

    @property
    def speed_category(self) -> str:
        """Backwards compatibility alias for activity category."""
        return self.activity_category

    @property
    def total_distance_m(self) -> float:
        """Backwards compatibility alias for cumulative distance."""
        return self.cumulative_distance_m


MAX_REALISTIC_SPEED_KMH = 38.0
MIN_SPEED_THRESHOLD_KMH = 2.0


def classify_speed_category(speed_kmh: float) -> str:
    """Categorize speed into standard FIFA physical performance categories."""
    if speed_kmh < 2.0:
        return "Standing"
    elif speed_kmh < 7.2:
        return "Walking"
    elif speed_kmh < 14.4:
        return "Jogging"
    elif speed_kmh < 19.8:
        return "Running"
    else:
        return "Sprinting"


class SpeedEstimator:
    """
    Physical velocity and distance estimation engine for football players.
    Uses Least-Squares Linear Regression with Outlier Residual Rejection
    to cancel leg oscillation jitter and produce continuous, accurate speeds.
    """

    def __init__(
        self,
        fps: float = 25.0,
        window_size: int = 18,
        min_frames_for_speed: int = 4,
        max_speed_kmh: float = 38.0,
        min_speed_kmh: float = 2.0,
        ema_alpha: float = 0.75,
    ):
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 0.04
        self.window_size = window_size
        self.min_frames_for_speed = min_frames_for_speed
        self.max_speed_kmh = max_speed_kmh
        self.min_speed_kmh = min_speed_kmh
        self.ema_alpha = ema_alpha

        # Position history: {track_id: deque([(t, x_m, y_m), ...], maxlen=window_size)}
        self.positions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window_size))
        
        # Stored metrics per track
        self.metrics: Dict[int, PlayerMetrics] = {}

    @property
    def players(self) -> Dict[int, PlayerMetrics]:
        """Dictionary of active player metrics."""
        return self.metrics

    def _classify_activity(self, speed_kmh: float) -> str:
        return classify_speed_category(speed_kmh)

    def _fit_velocity_regression(self, points: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        """
        Fit linear regression X(t) = v_x * t + x0 and Y(t) = v_y * t + y0 with outlier rejection.
        """
        if len(points) < self.min_frames_for_speed:
            return 0.0, 0.0

        pts_arr = np.array(points, dtype=np.float64)
        t = pts_arr[:, 0]
        x = pts_arr[:, 1]
        y = pts_arr[:, 2]

        t_mean = np.mean(t)
        t_zeroed = t - t_mean

        denom = np.sum(t_zeroed ** 2)
        if denom < 1e-6:
            return 0.0, 0.0

        # Initial slope
        vx = np.sum(t_zeroed * (x - np.mean(x))) / denom
        vy = np.sum(t_zeroed * (y - np.mean(y))) / denom

        # Outlier residual check (reject single-frame teleportation jumps > 3 sigma)
        pred_x = vx * t_zeroed + np.mean(x)
        pred_y = vy * t_zeroed + np.mean(y)
        resids = np.sqrt((x - pred_x) ** 2 + (y - pred_y) ** 2)
        med_resid = np.median(resids)

        inliers = resids <= max(1.5, med_resid * 3.0)
        if np.sum(inliers) >= 4:
            t_in = t_zeroed[inliers]
            denom_in = np.sum(t_in ** 2)
            if denom_in > 1e-6:
                vx = np.sum(t_in * (x[inliers] - np.mean(x[inliers]))) / denom_in
                vy = np.sum(t_in * (y[inliers] - np.mean(y[inliers]))) / denom_in

        return float(vx), float(vy)

    def update(
        self,
        track_ids: np.ndarray,
        positions_m: np.ndarray,
        frame_idx: int,
    ) -> Dict[int, PlayerMetrics]:
        """
        Update player position buffers and compute regression velocities and distances.
        """
        num_items = len(track_ids)
        current_time = frame_idx * self.dt

        for i in range(num_items):
            tid = int(track_ids[i])
            if tid < 0:
                continue

            xm, ym = float(positions_m[i, 0]), float(positions_m[i, 1])

            # Sanity check: Ignore points outside FIFA pitch boundaries
            if not (-5.0 <= xm <= 110.0 and -5.0 <= ym <= 73.0):
                continue

            hist = self.positions[tid]
            
            # Ensure PlayerMetrics object exists
            if tid not in self.metrics:
                self.metrics[tid] = PlayerMetrics(track_id=tid)

            # Cumulative distance calculation
            if len(hist) > 0:
                prev_t, prev_x, prev_y = hist[-1]
                step_dist = np.sqrt((xm - prev_x) ** 2 + (ym - prev_y) ** 2)
                dt_step = current_time - prev_t

                # Filter out impossible teleportation noise (> 12 m/s step)
                if dt_step > 0 and (step_dist / dt_step) <= (self.max_speed_kmh / 3.6):
                    if step_dist >= 0.01:
                        self.metrics[tid].cumulative_distance_m += step_dist

            hist.append((current_time, xm, ym))

            # Linear regression velocity
            vx, vy = self._fit_velocity_regression(list(hist))
            speed_ms = np.sqrt(vx ** 2 + vy ** 2)
            speed_kmh = speed_ms * 3.6

            # Apply speed clamping and stationary deadband
            if speed_kmh < self.min_speed_kmh:
                speed_kmh = 0.0
            elif speed_kmh > self.max_speed_kmh:
                speed_kmh = self.max_speed_kmh

            pm = self.metrics[tid]
            pm.current_speed_kmh = round(speed_kmh, 1)
            pm.smoothed_speed_kmh = round(
                self.ema_alpha * pm.smoothed_speed_kmh + (1.0 - self.ema_alpha) * speed_kmh, 1
            )
            if speed_kmh > pm.top_speed_kmh:
                pm.top_speed_kmh = round(speed_kmh, 1)
            pm.activity_category = self._classify_activity(pm.smoothed_speed_kmh)

        return self.metrics

    def get_team_summary(self) -> Dict[str, float]:
        """Aggregate total physical telemetry across all players."""
        if not self.metrics:
            return {"max_speed_kmh": 0.0, "total_distance_km": 0.0, "avg_speed_kmh": 0.0, "active_players_tracked": 0}

        top_speeds = [m.top_speed_kmh for m in self.metrics.values()]
        total_dist_m = sum(m.cumulative_distance_m for m in self.metrics.values())
        curr_speeds = [m.smoothed_speed_kmh for m in self.metrics.values() if m.smoothed_speed_kmh > 0]

        return {
            "max_speed_kmh": round(float(np.max(top_speeds)), 1) if top_speeds else 0.0,
            "total_distance_km": round(float(total_dist_m / 1000.0), 3),
            "avg_speed_kmh": round(float(np.mean(curr_speeds)), 1) if curr_speeds else 0.0,
            "active_players_tracked": len(self.metrics),
        }
