"""
Unit Tests for Camera Cut Detection Module (Milestone 10).
Validates HSV histogram dissimilarity, edge change detection, shot boundary triggers,
temporal debouncing, and optical flow failure penalties.
"""

import cv2
import numpy as np
import pytest

from src.tracking.cut_detector import CameraCutDetector, CutDetectionResult


def create_synthetic_pitch_frame(width: int = 640, height: int = 360, shift: int = 0) -> np.ndarray:
    """Generate a synthetic football pitch frame with green grass and white markings."""
    frame = np.full((height, width, 3), (35, 140, 45), dtype=np.uint8)  # Grass green
    # Draw touchline & penalty box lines
    cv2.line(frame, (50 + shift, 100), (width - 50 + shift, 100), (255, 255, 255), 3)
    cv2.line(frame, (50 + shift, 300), (width - 50 + shift, 300), (255, 255, 255), 3)
    cv2.circle(frame, (width // 2 + shift, height // 2), 60, (255, 255, 255), 3)
    return frame


def create_synthetic_replay_frame(width: int = 640, height: int = 360) -> np.ndarray:
    """Generate a synthetic close-up or dugout scene with completely different color profile."""
    frame = np.full((height, width, 3), (180, 40, 20), dtype=np.uint8)  # Deep blue/red
    # Draw different structural patterns
    cv2.rectangle(frame, (100, 50), (width - 100, height - 50), (20, 200, 220), -1)
    return frame


def test_camera_cut_detector_initialization():
    detector = CameraCutDetector(hist_threshold=0.50, edge_threshold=0.45, min_cut_interval=15)
    assert detector.hist_threshold == 0.50
    assert detector.edge_threshold == 0.45
    assert detector.min_cut_interval == 15
    assert detector.cut_count == 0
    assert detector.prev_hist is None


def test_camera_cut_on_identical_frames():
    detector = CameraCutDetector()
    frame = create_synthetic_pitch_frame()

    # First frame (initializes history)
    res1 = detector.detect_cut(frame, frame_idx=0)
    assert res1.is_cut is False
    assert res1.cut_count == 0

    # Second identical frame
    res2 = detector.detect_cut(frame, frame_idx=1)
    assert res2.is_cut is False
    assert res2.hist_distance < 0.05
    assert res2.edge_diff < 0.05
    assert res2.cut_count == 0


def test_camera_cut_on_smooth_panning_frames():
    detector = CameraCutDetector()

    # Feed slight panning frames (continuous video)
    for idx in range(10):
        frame = create_synthetic_pitch_frame(shift=idx * 2)
        res = detector.detect_cut(frame, frame_idx=idx)
        assert res.is_cut is False
        assert res.cut_count == 0


def test_camera_cut_on_abrupt_scene_transition():
    detector = CameraCutDetector(hist_threshold=0.40, edge_threshold=0.35, combined_threshold=0.45)

    # Frame 0 to 4: Grass pitch
    for idx in range(5):
        frame_pitch = create_synthetic_pitch_frame(shift=idx)
        detector.detect_cut(frame_pitch, frame_idx=idx)

    # Frame 5: Sudden cut to close-up replay scene
    frame_replay = create_synthetic_replay_frame()
    cut_res = detector.detect_cut(frame_replay, frame_idx=5)

    assert cut_res.is_cut is True
    assert cut_res.cut_type == "HARD_CUT"
    assert cut_res.cut_count == 1
    assert cut_res.hist_distance > 0.40
    assert cut_res.edge_diff > 0.30


def test_camera_cut_debounce_interval():
    detector = CameraCutDetector(min_cut_interval=15)

    # Frame 0: Initial pitch
    detector.detect_cut(create_synthetic_pitch_frame(), frame_idx=0)

    # Frame 1: Hard cut 1
    cut1 = detector.detect_cut(create_synthetic_replay_frame(), frame_idx=1)
    assert cut1.is_cut is True
    assert detector.cut_count == 1

    # Frame 2: Another abrupt shift immediately within debounce window (should be ignored)
    cut2 = detector.detect_cut(create_synthetic_pitch_frame(), frame_idx=2)
    assert cut2.is_cut is False
    assert detector.cut_count == 1

    # Frame 20: Cut occurring after debounce interval (should trigger cut 2)
    cut3 = detector.detect_cut(create_synthetic_replay_frame(), frame_idx=20)
    assert cut3.is_cut is True
    assert detector.cut_count == 2


def test_optical_flow_failure_penalty():
    detector = CameraCutDetector(combined_threshold=0.50)
    frame_a = create_synthetic_pitch_frame(shift=0)
    frame_b = create_synthetic_pitch_frame(shift=15)

    detector.detect_cut(frame_a, frame_idx=0)
    res_normal = detector.detect_cut(frame_b, frame_idx=1, flow_inlier_ratio=0.85)

    detector.reset()
    detector.detect_cut(frame_a, frame_idx=0)
    res_flow_failed = detector.detect_cut(frame_b, frame_idx=1, flow_inlier_ratio=0.05)

    assert res_flow_failed.combined_score > res_normal.combined_score
