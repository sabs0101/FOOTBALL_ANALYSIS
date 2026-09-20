"""
Unit Tests for Milestone 9: Camera Movement & Zoom Compensation (Global Motion Estimation).
"""

import cv2
import numpy as np
import pytest

from src.calibration.camera_motion import CameraMotionCompensator, CameraMotionResult
from src.detection.detector import DetectionResult


def create_synthetic_textured_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """Create a synthetic football pitch texture with white lines and field patterns."""
    img = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)  # Forest green

    # Add white field lines and grid markings
    cv2.rectangle(img, (50, 50), (width - 50, height - 50), (255, 255, 255), 4)
    cv2.line(img, (width // 2, 50), (width // 2, height - 50), (255, 255, 255), 3)
    cv2.circle(img, (width // 2, height // 2), 50, (255, 255, 255), 3)
    cv2.circle(img, (width // 2, height // 2), 4, (255, 255, 255), -1)

    # Add corner arcs and penalty boxes
    cv2.rectangle(img, (50, 100), (150, height - 100), (255, 255, 255), 2)
    cv2.rectangle(img, (width - 150, 100), (width - 50, height - 100), (255, 255, 255), 2)

    # Add grass striped lawn bands for feature rich optical flow
    for x in range(0, width, 40):
        cv2.line(img, (x, 0), (x, height), (30, 120, 30), 1)

    return img


def test_camera_motion_compensator_initialization():
    """Test compensator initialization and parameter defaults."""
    cmc = CameraMotionCompensator(
        max_features=300,
        quality_level=0.02,
        ransac_threshold=3.0,
    )
    assert cmc.max_features == 300
    assert cmc.quality_level == 0.02
    assert cmc.prev_gray is None


def test_camera_motion_estimation_on_synthetic_translation():
    """Test GME on pure synthetic camera translation/pan."""
    cmc = CameraMotionCompensator()
    frame1 = create_synthetic_textured_frame()

    # Frame 1 (initial frame) -> Returns identity
    res1 = cmc.estimate_motion(frame1, frame_idx=0)
    assert res1.is_valid
    assert abs(res1.dx_pixels) < 0.1
    assert abs(res1.dy_pixels) < 0.1

    # Frame 2: Shift image by +15px in X (simulates camera panning left)
    M_trans = np.float32([[1, 0, 15], [0, 1, 0]])
    frame2 = cv2.warpAffine(frame1, M_trans, (640, 360))

    res2 = cmc.estimate_motion(frame2, frame_idx=1)
    assert res2.is_valid
    assert abs(res2.dx_pixels - 15.0) < 2.0
    assert abs(res2.dy_pixels) < 2.0
    assert res2.pan_direction == "PAN LEFT"
    assert abs(res2.zoom_factor - 1.0) < 0.02


def test_camera_motion_estimation_on_synthetic_zoom():
    """Test GME on synthetic camera zoom scaling."""
    cmc = CameraMotionCompensator()
    frame1 = create_synthetic_textured_frame()

    cmc.estimate_motion(frame1, frame_idx=0)

    # Scale image around center by 1.03x (Zoom in)
    h, w = frame1.shape[:2]
    center = (w / 2.0, h / 2.0)
    M_zoom = cv2.getRotationMatrix2D(center, angle=0.0, scale=1.03)
    frame2 = cv2.warpAffine(frame1, M_zoom, (w, h))

    res2 = cmc.estimate_motion(frame2, frame_idx=1)
    assert res2.is_valid
    assert abs(res2.zoom_factor - 1.03) < 0.02
    assert res2.zoom_state == "ZOOM IN"


def test_foreground_player_masking():
    """Test that foreground player bounding boxes are masked out in the background mask."""
    cmc = CameraMotionCompensator()
    dummy_detections = DetectionResult(
        xyxy=np.array([[100, 100, 150, 200]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        class_ids=np.array([0], dtype=int),
        class_names=["person"],
    )

    mask = cmc._create_background_mask((360, 640), dummy_detections, margin=10)
    # Area inside bounding box should be 0 (masked out)
    assert mask[120, 120] == 0
    # Area outside bounding box should be 255 (background pitch)
    assert mask[300, 300] == 255


def test_warp_points_and_boxes():
    """Test warping points and bounding boxes by an affine camera transform."""
    transform = np.array([
        [1.0, 0.0, 10.0],
        [0.0, 1.0, -5.0],
    ], dtype=np.float32)

    pts = np.array([[100.0, 200.0], [50.0, 50.0]], dtype=np.float32)
    warped_pts = CameraMotionCompensator.warp_points(pts, transform)

    assert abs(warped_pts[0, 0] - 110.0) < 1e-4
    assert abs(warped_pts[0, 1] - 195.0) < 1e-4

    boxes = np.array([[100.0, 100.0, 150.0, 200.0]], dtype=np.float32)
    warped_boxes = CameraMotionCompensator.warp_boxes(boxes, transform)
    assert abs(warped_boxes[0, 0] - 110.0) < 1e-4
    assert abs(warped_boxes[0, 1] - 95.0) < 1e-4
    assert abs(warped_boxes[0, 2] - 160.0) < 1e-4
    assert abs(warped_boxes[0, 3] - 195.0) < 1e-4
