"""
Unit Tests for Milestone 8: Ball Tracking, Kalman Filtering, Trajectory Interpolation,
and Player Possession Assignment.
"""

import numpy as np
import pytest

from src.detection.detector import DetectionResult
from src.tracking.ball_tracker import BallKalmanFilter, BallState, BallTracker, PossessionResult


def create_mock_detection(
    ball_xyxy: np.ndarray = None,
    ball_conf: float = 0.85,
    frame_idx: int = 0,
) -> DetectionResult:
    """Helper to create synthetic DetectionResult with optional ball box."""
    if ball_xyxy is not None:
        xyxy = np.array([ball_xyxy], dtype=np.float32)
        confidences = np.array([ball_conf], dtype=np.float32)
        class_ids = np.array([32], dtype=int)
        class_names = ["sports_ball"]
        tracker_ids = np.array([-1], dtype=int)
    else:
        xyxy = np.empty((0, 4), dtype=np.float32)
        confidences = np.empty((0,), dtype=np.float32)
        class_ids = np.empty((0,), dtype=int)
        class_names = []
        tracker_ids = np.empty((0,), dtype=int)

    return DetectionResult(
        xyxy=xyxy,
        confidences=confidences,
        class_ids=class_ids,
        class_names=class_names,
        tracker_ids=tracker_ids,
        frame_idx=frame_idx,
    )


def test_ball_kalman_filter_state_update():
    """Test Kalman filter initialization, prediction, and state convergence."""
    kf = BallKalmanFilter(dt=0.04, process_noise_std=8.0, meas_noise_std=0.8)
    assert not kf.is_initialized

    kf.initialize(100.0, 200.0, vx=50.0, vy=0.0)
    assert kf.is_initialized

    pred_x, pred_y = kf.predict()
    assert pred_x > 100.0  # Moves forward due to vx=50.0

    upd_x, upd_y = kf.update(103.0, 201.0)
    assert abs(upd_x - 103.0) < 2.0
    assert abs(upd_y - 201.0) < 2.0


def test_ball_tracker_initialization():
    """Test BallTracker parameters, thresholds, and initial state."""
    tracker = BallTracker(
        fps=25.0,
        max_interpolation_gap=6,
        max_speed_kmh=140.0,
        possession_radius_m=1.8,
    )
    assert tracker.fps == 25.0
    assert tracker.max_interpolation_gap == 6
    assert tracker.possession_radius_m == 1.8
    assert tracker.last_valid_state is None
    assert tracker.current_possessor_id is None
    assert tracker.turnover_count == 0


def test_ball_tracker_outlier_rejection():
    """Test that sudden unrealistic teleportations (>380px or >140km/h) are rejected."""
    tracker = BallTracker(fps=25.0, max_speed_kmh=140.0)

    # Frame 0: Valid detection at (500, 500)
    det0 = create_mock_detection(np.array([490, 490, 510, 510]), frame_idx=0)
    ball_state, _ = tracker.update(det0, frame_idx=0)
    assert ball_state is not None
    assert not ball_state.is_interpolated
    assert abs(ball_state.center_px[0] - 500.0) < 1.0

    # Frame 1: Teleportation to (1500, 1500) -> Outlier rejected
    det1 = create_mock_detection(np.array([1490, 1490, 1510, 1510]), frame_idx=1)
    ball_state_outlier, _ = tracker.update(det1, frame_idx=1)
    # Outlier rejected -> Falls back to Kalman prediction
    assert ball_state_outlier is not None
    assert ball_state_outlier.is_interpolated  # Interpolated rather than jumping to 1500
    assert abs(ball_state_outlier.center_px[0] - 500.0) < 50.0


def test_ball_tracker_gap_interpolation():
    """Test trajectory interpolation across temporary missing frames (dropouts)."""
    tracker = BallTracker(fps=25.0, max_interpolation_gap=4)

    # Frames 0..2: Ball moving linearly at ~20px per frame in x
    for f in range(3):
        x = 200.0 + f * 20.0
        det = create_mock_detection(np.array([x - 5, 300, x + 5, 310]), frame_idx=f)
        state, _ = tracker.update(det, frame_idx=f)
        assert state is not None
        assert not state.is_interpolated

    # Frames 3..6: Detection missing (occluded by player, 4 frames) -> Interpolated
    for f in range(3, 7):
        det_empty = create_mock_detection(None, frame_idx=f)
        state_interp, _ = tracker.update(det_empty, frame_idx=f)
        assert state_interp is not None
        assert state_interp.is_interpolated
        assert state_interp.center_px[0] > 230.0  # Trajectory projected forward

    # Frame 7: 5th consecutive lost frame (> max_interpolation_gap=4) -> Ball lost
    det_empty = create_mock_detection(None, frame_idx=7)
    state_lost, _ = tracker.update(det_empty, frame_idx=7)
    assert state_lost is None


def test_possession_assignment_proximity_and_turnover():
    """Test on-ball possession attribution to closest player, debounce, and turnover tracking."""
    tracker = BallTracker(fps=25.0, possession_radius_m=1.8, hysteresis_frames=2)

    # Synthetic players in metric pitch coordinates (X, Y)
    player_positions = np.array([
        [50.0, 30.0],  # Player #10 (Team A: 0)
        [65.0, 40.0],  # Player #19 (Team B: 1)
        [20.0, 15.0],  # Player #5  (Team A: 0)
    ], dtype=np.float32)
    player_track_ids = np.array([10, 19, 5], dtype=int)
    player_team_ids = np.array([0, 1, 0], dtype=int)

    # 1. Ball at (50.5, 30.2) -> Distance to Player #10 is ~0.54m (within 1.8m)
    mock_state = BallState(frame_idx=0, position_m=(50.5, 30.2), center_px=(500, 500))
    possession_res = tracker._assign_possession(
        mock_state,
        player_positions,
        player_track_ids,
        player_team_ids,
        frame_idx=0,
    )
    assert possession_res.possessing_player_id == 10
    assert possession_res.possessing_team_id == 0
    assert possession_res.possessing_team_name == "Team A"
    assert possession_res.distance_to_ball_m <= 1.0

    # 2. Frame 1: Ball passes to Player #19 at (64.8, 39.9) -> Distance ~0.22m (Immediate takeover < 1.05m)
    mock_state_p19 = BallState(frame_idx=1, position_m=(64.8, 39.9), center_px=(700, 400))
    possession_res_turnover = tracker._assign_possession(
        mock_state_p19,
        player_positions,
        player_track_ids,
        player_team_ids,
        frame_idx=1,
    )
    assert possession_res_turnover.possessing_player_id == 19
    assert possession_res_turnover.possessing_team_id == 1
    assert possession_res_turnover.possessing_team_name == "Team B"
    assert possession_res_turnover.turnover_count == 1  # Team A -> Team B turnover registered!


def test_contested_ball_detection():
    """Test 50-50 contested ball when opposing players are equidistant to the ball."""
    tracker = BallTracker(fps=25.0, possession_radius_m=2.0)

    # Two opposing players contesting the ball at (50.0, 30.0)
    player_positions = np.array([
        [49.5, 30.0],  # Player #7 (Team A: 0) - distance 0.5m
        [50.6, 30.0],  # Player #14 (Team B: 1) - distance 0.6m
    ], dtype=np.float32)
    player_track_ids = np.array([7, 14], dtype=int)
    player_team_ids = np.array([0, 1], dtype=int)

    mock_state = BallState(frame_idx=0, position_m=(50.0, 30.0), center_px=(500, 500))
    possession_res = tracker._assign_possession(
        mock_state,
        player_positions,
        player_track_ids,
        player_team_ids,
        frame_idx=0,
    )
    assert possession_res.is_contested
    assert possession_res.possessing_team_name == "Contested 50-50"


def test_possession_array_bounds_safety():
    """Test that possession assignment gracefully handles when non-player detections or mismatch arrays are passed."""
    tracker = BallTracker(fps=25.0, possession_radius_m=2.0)

    # 26 positions (e.g. 25 players + 1 referee), but only 25 player track/team IDs
    positions = np.zeros((26, 2), dtype=np.float32)
    positions[25] = [50.0, 30.0]  # Closest to ball is index 25 (e.g. non-player)
    player_track_ids = np.arange(25, dtype=int)
    player_team_ids = np.zeros(25, dtype=int)

    mock_state = BallState(frame_idx=0, position_m=(50.0, 30.0), center_px=(500, 500))
    # Should not raise IndexError: index 25 is out of bounds for axis 0 with size 25
    possession_res = tracker._assign_possession(
        mock_state,
        positions,
        player_track_ids,
        player_team_ids,
        frame_idx=0,
    )
    assert possession_res is not None
