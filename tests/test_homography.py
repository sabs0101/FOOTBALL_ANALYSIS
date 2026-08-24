"""
Unit tests for Homography Estimation, Pitch Template, and 2D Tactical Radar (Milestone 4).
"""

import numpy as np
import pytest
from src.calibration.homography import PitchHomography, HomographyResult
from src.calibration.template import (
    PitchTemplate,
    PITCH_KEYPOINTS_FIFA,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from src.visualization.radar import TacticalRadar


def test_pitch_template_dimensions():
    template = PitchTemplate()
    assert template.length_m == 105.0
    assert template.width_m == 68.0
    assert "center_spot" in PITCH_KEYPOINTS_FIFA
    assert PITCH_KEYPOINTS_FIFA["center_spot"] == (52.5, 34.0)


def test_homography_computation():
    """
    Test homography calculation on 4 canonical point correspondences.
    """
    calibrator = PitchHomography()

    src_pts = np.array([
        [100.0, 200.0],
        [900.0, 200.0],
        [1100.0, 800.0],
        [0.0, 800.0],
    ], dtype=np.float32)

    dst_pts = np.array([
        [0.0, 68.0],
        [105.0, 68.0],
        [105.0, 0.0],
        [0.0, 0.0],
    ], dtype=np.float32)

    result = calibrator.compute_homography(src_pts, dst_pts)

    assert isinstance(result, HomographyResult)
    assert result.is_valid
    assert result.H.shape == (3, 3)
    assert result.H_inv.shape == (3, 3)
    assert result.reprojection_error < 0.10


def test_point_projection_roundtrip():
    """
    Verify mathematical round-trip consistency:
    Image (u, v) -> Pitch (X, Y) -> Image (u, v)
    """
    calibrator = PitchHomography()

    src_pts = np.array([
        [200.0, 250.0],
        [800.0, 250.0],
        [950.0, 750.0],
        [50.0, 750.0],
    ], dtype=np.float32)

    dst_pts = np.array([
        [0.0, 68.0],
        [105.0, 68.0],
        [105.0, 0.0],
        [0.0, 0.0],
    ], dtype=np.float32)

    res = calibrator.compute_homography(src_pts, dst_pts)

    # Test query points in image space
    query_img_pts = np.array([
        [500.0, 500.0],
        [300.0, 400.0],
    ], dtype=np.float32)

    pitch_pts = calibrator.image_to_pitch(query_img_pts, res.H)
    assert pitch_pts.shape == (2, 2)

    # Invert back to image space
    reprojected = calibrator.pitch_to_image(pitch_pts, res.H_inv)
    assert reprojected.shape == (2, 2)

    # Error should be less than 0.01 pixels
    diff = np.max(np.abs(query_img_pts - reprojected))
    assert diff < 0.01, f"Roundtrip error too high: {diff}"


def test_tactical_radar_rendering():
    """
    Verify that 2D radar renders without errors and outputs valid image shape.
    """
    radar = TacticalRadar(radar_width=320, radar_height=200)

    # Synthetic player positions in real meters
    player_positions = np.array([
        [52.5, 34.0],  # Center
        [10.0, 34.0],  # Left
        [95.0, 34.0],  # Right
    ], dtype=np.float32)

    player_ids = np.array([7, 10, 11], dtype=np.int32)
    ball_pos = (50.0, 32.0)

    img = radar.render_radar(
        player_positions_m=player_positions,
        player_track_ids=player_ids,
        ball_position_m=ball_pos,
    )

    assert img.shape == (200, 320, 3)
    assert np.mean(img) > 0
