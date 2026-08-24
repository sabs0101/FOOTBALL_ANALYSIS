"""
Unit and integration tests for Football Pitch & Field Line Detection (Milestone 3).
"""

import cv2
import numpy as np
import pytest
from src.detection.detector import DetectionResult
from src.pitch.detector import PitchDetector, PitchResult


def test_pitch_detector_initialization():
    detector = PitchDetector(
        hsv_green_lower=(30, 40, 40),
        hsv_green_upper=(90, 255, 255),
        morph_kernel_size=5,
    )
    assert detector.min_pitch_area_ratio == 0.20
    assert detector.white_contrast_thresh == 18


def test_pitch_mask_on_synthetic_grass_field():
    """
    Generate synthetic 720p image:
    - Top 25% is gray/crowd (BGR 100, 100, 100)
    - Bottom 75% is green football pitch (BGR 34, 139, 34)
    """
    h, w = 720, 1280
    frame = np.full((h, w, 3), (100, 100, 100), dtype=np.uint8)
    pitch_y_start = int(h * 0.25)
    frame[pitch_y_start:h, :] = (34, 139, 34)

    detector = PitchDetector()
    mask, hull, area_ratio = detector.detect_mask(frame)

    assert mask.shape == (h, w)
    assert area_ratio >= 0.50
    assert np.mean(mask[0:int(h * 0.15), :]) == 0
    assert np.mean(mask[int(h * 0.40):int(h * 0.75), :]) == 255


def test_line_detection_on_synthetic_pitch():
    """
    Generate green pitch with clean white horizontal and vertical field lines.
    Verify that line detection identifies the lines.
    """
    h, w = 600, 800
    frame = np.full((h, w, 3), (40, 140, 40), dtype=np.uint8)

    # Draw white pitch lines
    cv2.line(frame, (100, 100), (700, 100), (255, 255, 255), 4)  # Top line
    cv2.line(frame, (100, 500), (700, 500), (255, 255, 255), 4)  # Bottom line
    cv2.line(frame, (400, 100), (400, 500), (255, 255, 255), 4)  # Halfway line

    detector = PitchDetector(white_contrast_thresh=15, min_line_length=30)
    result = detector.detect_lines(frame)

    assert isinstance(result, PitchResult)
    assert len(result.lines) > 0, "Expected at least 1 detected field line"
    assert result.pitch_area_ratio > 0.60


def test_crowd_filtering():
    """
    Verify that detections outside playable field (in stands or dugout) are filtered out,
    while on-pitch players and the football are preserved.
    """
    h, w = 720, 1280
    pitch_mask = np.zeros((h, w), dtype=np.uint8)
    pitch_mask[200:600, :] = 255  # Playable pitch between y=200 and y=600

    detector = PitchDetector()

    detections = DetectionResult(
        xyxy=np.array([
            [500, 20, 550, 100],   # Crowd spectator (y=100) -> Filtered
            [600, 350, 650, 450],  # On-pitch player (y=450) -> Kept
            [700, 620, 750, 690],  # Coach in dugout (y=690) -> Filtered
            [400, 50, 420, 70],    # Ball -> Kept
        ], dtype=np.float32),
        confidences=np.array([0.80, 0.90, 0.88, 0.85], dtype=np.float32),
        class_ids=np.array([0, 0, 0, 32], dtype=np.int32),
        class_names=["person", "person", "person", "sports ball"],
        frame_idx=0,
    )

    filtered = detector.filter_detections_on_pitch(detections, pitch_mask)

    assert len(filtered.xyxy) == 2, f"Expected 2 detections (player + ball), got {len(filtered.xyxy)}"
    assert filtered.class_ids[0] == 0   # On-pitch player kept
    assert filtered.class_ids[1] == 32  # Ball kept
    assert filtered.xyxy[0][3] == 450   # Kept player is the on-pitch player
