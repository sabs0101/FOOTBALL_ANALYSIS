"""
Unit tests for Physical Kinematics, Speed Estimation, and Distance Tracking (Milestone 5).
"""

import numpy as np
import pytest
from src.analytics.speed_distance import (
    SpeedEstimator,
    PlayerMetrics,
    MAX_REALISTIC_SPEED_KMH,
    classify_speed_category,
)


def test_speed_estimator_initialization():
    estimator = SpeedEstimator(fps=25.0, window_size=15)
    assert estimator.fps == 25.0
    assert estimator.dt == 0.04
    assert estimator.max_speed_kmh == MAX_REALISTIC_SPEED_KMH
    assert len(estimator.players) == 0


def test_speed_category_classification():
    assert classify_speed_category(1.5) == "Standing"
    assert classify_speed_category(5.0) == "Walking"
    assert classify_speed_category(10.5) == "Jogging"
    assert classify_speed_category(18.0) == "Running"
    assert classify_speed_category(28.5) == "Sprinting"


def test_constant_velocity_regression_speed():
    """
    Simulate a player running at 5.0 m/s (18.0 km/h) for 25 frames (1 second).
    Verify regression speed converges exactly to 18.0 km/h.
    """
    fps = 25.0
    estimator = SpeedEstimator(fps=fps, window_size=15, min_frames_for_speed=6)
    track_ids = np.array([10], dtype=np.int32)

    start_x = 20.0
    start_y = 34.0

    for f in range(25):
        curr_x = start_x + f * 0.20  # 0.20m per frame = 5.0 m/s = 18.0 km/h
        pos = np.array([[curr_x, start_y]], dtype=np.float32)
        metrics = estimator.update(track_ids, pos, frame_idx=f)

    p10 = metrics[10]
    assert isinstance(p10, PlayerMetrics)
    assert pytest.approx(p10.current_speed_kmh, rel=0.05) == 18.0
    assert p10.speed_category == "Running"


def test_cumulative_distance_accumulation():
    """
    Verify distance accumulates accurately over 250 frames.
    """
    fps = 25.0
    estimator = SpeedEstimator(fps=fps, window_size=15)
    track_ids = np.array([7], dtype=np.int32)

    total_frames = 250
    step = 0.20  # 0.20m per frame

    for f in range(total_frames):
        curr_pos = np.array([[10.0 + f * step, 20.0]], dtype=np.float32)
        metrics = estimator.update(track_ids, curr_pos, frame_idx=f)

    p7 = metrics[7]
    assert pytest.approx(p7.total_distance_m, rel=0.05) == (total_frames - 1) * step


def test_unrealistic_speed_teleportation_clamping():
    """
    Verify that an impossible jump is rejected and does not corrupt total distance.
    """
    fps = 25.0
    estimator = SpeedEstimator(fps=fps, window_size=15, max_speed_kmh=38.0)
    track_ids = np.array([9], dtype=np.int32)

    estimator.update(track_ids, np.array([[20.0, 20.0]], dtype=np.float32), frame_idx=0)
    metrics = estimator.update(track_ids, np.array([[20.2, 20.0]], dtype=np.float32), frame_idx=1)
    dist_normal = metrics[9].total_distance_m
    assert pytest.approx(dist_normal, rel=0.05) == 0.2

    metrics = estimator.update(track_ids, np.array([[80.0, 20.0]], dtype=np.float32), frame_idx=2)
    assert metrics[9].total_distance_m == dist_normal


def test_team_summary_aggregation():
    """
    Verify aggregate physical analytics summary across multiple players.
    """
    fps = 25.0
    estimator = SpeedEstimator(fps=fps)

    track_ids = np.array([1, 2], dtype=np.int32)
    pos = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    estimator.update(track_ids, pos, frame_idx=0)

    pos_step = np.array([[10.2, 10.0], [20.3, 20.0]], dtype=np.float32)
    estimator.update(track_ids, pos_step, frame_idx=1)

    summary = estimator.get_team_summary()
    assert "total_distance_km" in summary
    assert "max_speed_kmh" in summary
    assert "avg_speed_kmh" in summary
    assert summary["active_players_tracked"] == 2
