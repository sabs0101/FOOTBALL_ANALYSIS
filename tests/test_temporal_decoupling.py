"""
Unit Tests for Milestone 14: Temporal Detector Decoupling, Sparse GME & Homography Propagation.
"""

import numpy as np
import pytest
import cv2

from src.calibration.camera_motion import CameraMotionCompensator, CameraMotionResult
from src.calibration.homography import PitchHomography, HomographyResult
from src.detection.detector import DetectionResult
from src.tracking.tracker import PlayerTracker


def create_textured_pitch_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a high-contrast synthetic pitch texture with field markings and corner features."""
    img = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)  # Grass green

    # Field border and halfway line
    cv2.rectangle(img, (80, 80), (width - 80, height - 80), (255, 255, 255), 4)
    cv2.line(img, (width // 2, 80), (width // 2, height - 80), (255, 255, 255), 4)
    cv2.circle(img, (width // 2, height // 2), 90, (255, 255, 255), 4)
    cv2.circle(img, (width // 2, height // 2), 6, (255, 255, 255), -1)

    # Penalty areas and penalty spots
    cv2.rectangle(img, (80, 180), (280, height - 180), (255, 255, 255), 3)
    cv2.rectangle(img, (width - 280, 180), (width - 80, height - 180), (255, 255, 255), 3)
    cv2.circle(img, (180, height // 2), 5, (255, 255, 255), -1)
    cv2.circle(img, (width - 180, height // 2), 5, (255, 255, 255), -1)

    # Text markings and patterns
    for x in range(120, width - 120, 60):
        for y in range(120, height - 120, 60):
            cv2.circle(img, (x, y), 3, (240, 240, 240), -1)

    return img


class TestSparseGMEAndHomographyPropagation:
    """Test suite for Sparse High-Speed GME and Homography Matrix Motion Propagation."""

    def test_sparse_gme_execution_speed_and_rescaling(self):
        comp = CameraMotionCompensator(sparse_mode=True, downscale_factor=0.5)

        # Create two 720p synthetic frames with a +16px horizontal camera shift
        frame1 = create_textured_pitch_frame(1280, 720)
        shift_x = 16.0
        shift_y = 6.0
        M_gt = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame2 = cv2.warpAffine(frame1, M_gt, (1280, 720))

        # Initial frame initializes features
        res1 = comp.estimate_motion(frame1, frame_idx=0)
        assert res1.is_valid

        # Second frame estimates motion
        res2 = comp.estimate_motion(frame2, frame_idx=1)
        assert res2.is_valid
        # Due to 0.5 downscale, translation should be rescaled back close to +16.0, +6.0
        assert abs(res2.dx_pixels - shift_x) < 2.0
        assert abs(res2.dy_pixels - shift_y) < 2.0
        assert res2.pan_direction == "PAN LEFT"

    def test_homography_motion_propagation_accuracy(self):
        calibrator = PitchHomography()
        base_H_res = calibrator.estimate_broadcast_homography((720, 1280))
        assert base_H_res.is_valid

        # Simulate camera translation dx = 30px, dy = 10px
        transform_affine = np.array([
            [1.0, 0.0, 30.0],
            [0.0, 1.0, 10.0],
        ], dtype=np.float32)

        # Propagate homography to next frame
        prop_H_res = calibrator.propagate_homography(base_H_res, transform_affine)
        assert prop_H_res.is_valid
        assert prop_H_res.H.shape == (3, 3)
        assert prop_H_res.H_inv.shape == (3, 3)

        # Test point consistency: A point p_1 in frame 1 moves to p_2 = p_1 + [30, 10] in frame 2
        p1_img = np.array([[500.0, 400.0]], dtype=np.float32)
        p2_img = np.array([[530.0, 410.0]], dtype=np.float32)

        # Both points must project to the same world pitch coordinates
        world_p1 = calibrator.image_to_pitch(p1_img, base_H_res.H)
        world_p2 = calibrator.image_to_pitch(p2_img, prop_H_res.H)

        assert np.allclose(world_p1, world_p2, atol=0.05)


class TestTemporalDetectorDecoupling:
    """Test suite for PlayerTracker intervening frame propagation without neural detector calls."""

    def test_intervening_frame_propagation_state_continuity(self):
        tracker = PlayerTracker(frame_rate=25, trail_length=10)

        # Frame 0: Keyframe with 2 active players
        boxes_0 = np.array([[100.0, 200.0, 140.0, 280.0], [300.0, 400.0, 350.0, 500.0]], dtype=np.float32)
        confs_0 = np.array([0.90, 0.85], dtype=np.float32)
        cids_0 = np.array([0, 0], dtype=int)
        det_0 = DetectionResult(
            xyxy=boxes_0,
            confidences=confs_0,
            class_ids=cids_0,
            class_names=["person", "person"],
            frame_idx=0,
        )

        res_0 = tracker.update(det_0)
        assert len(res_0) == 2
        tids_0 = res_0.tracker_ids.copy()
        assert all(t >= 0 for t in tids_0)

        # Frame 1: Intervening frame (YOLO skipped, use propagate_intervening)
        # Simulate camera translation dx = 5px, dy = 0px
        camera_M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        res_1 = tracker.propagate_intervening(camera_transform=camera_M, frame_idx=1)

        assert len(res_1) == 2
        assert np.array_equal(res_1.tracker_ids, tids_0)
        # Verify bounding box was translated by +5px horizontally
        assert abs(res_1.xyxy[0, 0] - (boxes_0[0, 0] + 5.0)) < 1.0
        assert abs(res_1.xyxy[1, 0] - (boxes_0[1, 0] + 5.0)) < 1.0

        # Verify trails were updated
        assert len(res_1.trails[tids_0[0]]) == 2
        assert len(res_1.trails[tids_0[1]]) == 2

    def test_intervening_frame_with_camera_cut_reset(self):
        tracker = PlayerTracker(frame_rate=25)

        # Setup initial track
        boxes = np.array([[50.0, 50.0, 90.0, 130.0]], dtype=np.float32)
        det = DetectionResult(
            xyxy=boxes,
            confidences=np.array([0.95], dtype=np.float32),
            class_ids=np.array([0], dtype=int),
            class_names=["person"],
            frame_idx=0,
        )
        _ = tracker.update(det)

        # Reset on cut
        tracker.reset_tracks()
        # Propagate after cut with empty state
        res_after_cut = tracker.propagate_intervening(frame_idx=1)
        assert len(res_after_cut) == 0
