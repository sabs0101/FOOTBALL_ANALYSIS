"""
Unit and integration tests for Player Detection pipeline.
"""

import numpy as np
import pytest
from src.detection.detector import PlayerDetector, DetectionResult
from src.utils.config import load_config, get_device, validate_config
from src.visualization.annotator import VideoAnnotator


def test_config_loading():
    config = load_config("config.yaml")
    assert "detection" in config
    assert "system" in config
    assert "visualization" in config
    assert config["detection"]["conf_threshold"] > 0


def test_validate_config_passes_for_valid_config():
    config = load_config("config.yaml")
    # Should not raise for a well-formed config
    validate_config(config)


def test_validate_config_raises_on_invalid_threshold():
    config = load_config("config.yaml")
    config["detection"]["conf_threshold"] = 1.5  # invalid
    with pytest.raises(ValueError, match="conf_threshold"):
        validate_config(config)


def test_validate_config_raises_on_missing_section():
    config = {"detection": {"conf_threshold": 0.5}, "system": {}}
    # "visualization" section is missing
    with pytest.raises(KeyError, match="visualization"):
        validate_config(config)


def test_device_selection():
    device = get_device("auto")
    assert device is not None
    assert str(device) in ["cuda:0", "cpu"]


def test_detector_on_synthetic_frame():
    detector = PlayerDetector(
        model_name="yolov8n.pt",  # Use fast nano model for lightweight unit tests
        conf_threshold=0.25,
        device="cpu",  # CPU test guarantees CI/local stability
    )

    # Create a synthetic 720p green grass frame (BGR)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]  # Forest Green

    results = detector.detect(frame, frame_idx=0)
    assert isinstance(results, DetectionResult)
    assert results.xyxy.ndim == 2
    assert results.xyxy.shape[1] == 4 or results.xyxy.shape[0] == 0
    assert len(results.confidences) == len(results.xyxy)


def test_annotator_rendering():
    annotator = VideoAnnotator(draw_hud=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]

    dummy_boxes = np.array([[100, 100, 200, 300], [400, 400, 500, 600]], dtype=np.float32)
    dummy_conf = np.array([0.92, 0.85], dtype=np.float32)
    dummy_cls = np.array([0, 32], dtype=np.int32)
    dummy_names = ["person", "sports ball"]

    det = DetectionResult(
        xyxy=dummy_boxes,
        confidences=dummy_conf,
        class_ids=dummy_cls,
        class_names=dummy_names,
        frame_idx=1,
    )

    annotated = annotator.annotate(
        frame=frame,
        detections=det,
        fps=60.0,
        frame_idx=1,
        total_frames=100,
        device_name="RTX 4050",
    )

    assert annotated.shape == frame.shape
    # Ensure pixels were modified by annotations
    assert not np.array_equal(frame, annotated)
