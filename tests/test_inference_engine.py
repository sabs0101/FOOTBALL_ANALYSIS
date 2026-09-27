"""
Unit Tests for High-Performance GPU Inference Engine and Detector Backend (Milestone 13).
"""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from src.detection.detector import DetectionResult, PlayerDetector


class TestDetectionResult:
    """Test suite for DetectionResult container and convenience methods."""

    def test_detection_result_properties_and_alias(self):
        xyxy = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32)
        confs = np.array([0.92, 0.85], dtype=np.float32)
        cids = np.array([0, 32], dtype=int)
        cnames = ["person", "sports_ball"]
        tids = np.array([1, -1], dtype=int)

        res = DetectionResult(
            xyxy=xyxy,
            confidences=confs,
            class_ids=cids,
            class_names=cnames,
            tracker_ids=tids,
            frame_idx=5,
        )

        assert len(res) == 2
        assert res.num_detections == 2
        assert res.frame_idx == 5
        # Verify track_ids alias
        assert np.array_equal(res.track_ids, tids)

        # Test track_ids setter
        new_tids = np.array([99, 100], dtype=int)
        res.track_ids = new_tids
        assert np.array_equal(res.tracker_ids, new_tids)

    def test_filter_by_class(self):
        xyxy = np.array([[10, 20, 30, 40], [100, 100, 110, 110], [50, 60, 70, 80]], dtype=np.float32)
        confs = np.array([0.9, 0.5, 0.85], dtype=np.float32)
        cids = np.array([0, 32, 0], dtype=int)
        cnames = ["person", "sports_ball", "person"]
        tids = np.array([1, -1, 2], dtype=int)

        res = DetectionResult(
            xyxy=xyxy,
            confidences=confs,
            class_ids=cids,
            class_names=cnames,
            tracker_ids=tids,
        )

        players = res.get_players()
        assert len(players) == 2
        assert np.all(players.class_ids == 0)
        assert players.class_names == ["person", "person"]
        assert np.array_equal(players.tracker_ids, np.array([1, 2]))

        ball = res.get_ball()
        assert len(ball) == 1
        assert ball.class_ids[0] == 32
        assert ball.class_names == ["sports_ball"]

        refs = res.get_referees()
        assert len(refs) == 0

    def test_get_foot_positions(self):
        xyxy = np.array([[10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 120.0, 250.0]], dtype=np.float32)
        res = DetectionResult(
            xyxy=xyxy,
            confidences=np.array([0.9, 0.8], dtype=np.float32),
            class_ids=np.array([0, 0], dtype=int),
            class_names=["person", "person"],
        )

        feet = res.get_foot_positions()
        assert feet.shape == (2, 2)
        # Person 1 foot: x = (10+30)/2 = 20.0, y = 40.0
        assert feet[0, 0] == 20.0
        assert feet[0, 1] == 40.0
        # Person 2 foot: x = (100+120)/2 = 110.0, y = 250.0
        assert feet[1, 0] == 110.0
        assert feet[1, 1] == 250.0

        # Empty detections test
        empty_res = DetectionResult(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=int),
            class_names=[],
        )
        assert empty_res.get_foot_positions().shape == (0, 2)


class TestPlayerDetectorEngine:
    """Test suite for PlayerDetector acceleration, backend resolution, and fallback."""

    @patch("src.detection.detector.YOLO")
    def test_backend_resolution_and_metadata(self, mock_yolo):
        detector = PlayerDetector(
            model_name="yolov8n.pt",
            device="cpu",
            half=False,
            warmup_iterations=0,
        )
        info = detector.get_backend_info()
        assert "backend" in info
        assert info["device"] == "cpu"
        assert info["half"] is False
        assert info["imgsz"] == 1280

    @patch("src.detection.detector.YOLO")
    def test_coordinate_sanitization_and_clipping(self, mock_yolo):
        mock_instance = MagicMock()
        mock_yolo.return_value = mock_instance
        mock_instance.names = {0: "person", 32: "sports_ball"}

        # Return boxes that spill outside image frame (e.g. negative or > width/height)
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.xyxy.cpu().numpy.return_value = np.array([[-5.0, -10.0, 1300.0, 750.0]], dtype=np.float32)
        mock_boxes.conf.cpu().numpy.return_value = np.array([0.95], dtype=np.float32)
        mock_boxes.cls.cpu().numpy.return_value = np.array([0.0], dtype=np.float32)

        mock_pred = MagicMock()
        mock_pred.boxes = mock_boxes
        mock_instance.predict.return_value = [mock_pred]

        detector = PlayerDetector(model_name="yolov8n.pt", device="cpu", warmup_iterations=0)
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(dummy_frame)

        assert len(result) == 1
        # Coordinates must be clamped inside [0, 1279] and [0, 719]
        assert result.xyxy[0, 0] >= 0.0
        assert result.xyxy[0, 1] >= 0.0
        assert result.xyxy[0, 2] <= 1279.0
        assert result.xyxy[0, 3] <= 719.0

    @patch("src.detection.detector.YOLO")
    def test_dual_confidence_filtering(self, mock_yolo):
        mock_instance = MagicMock()
        mock_yolo.return_value = mock_instance
        mock_instance.names = {0: "person", 32: "sports_ball"}

        # Candidate detections:
        # Box 0: person with conf 0.15 (below 0.18 threshold -> should drop)
        # Box 1: person with conf 0.85 (above 0.18 threshold -> keep)
        # Box 2: ball with conf 0.14 (above 0.12 ball threshold -> keep)
        # Box 3: ball with conf 0.08 (below 0.12 ball threshold -> drop)
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 4
        mock_boxes.xyxy.cpu().numpy.return_value = np.array([
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [50, 50, 55, 55],
            [60, 60, 65, 65],
        ], dtype=np.float32)
        mock_boxes.conf.cpu().numpy.return_value = np.array([0.15, 0.85, 0.14, 0.08], dtype=np.float32)
        mock_boxes.cls.cpu().numpy.return_value = np.array([0, 0, 32, 32], dtype=np.float32)

        mock_pred = MagicMock()
        mock_pred.boxes = mock_boxes
        mock_instance.predict.return_value = [mock_pred]

        detector = PlayerDetector(
            model_name="yolov8n.pt",
            conf_threshold=0.18,
            conf_ball=0.12,
            device="cpu",
            warmup_iterations=0,
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(dummy_frame)

        assert len(result) == 2
        # Keep person with conf 0.85 and ball with conf 0.14
        assert 0 in result.class_ids
        assert 32 in result.class_ids

    @patch("src.detection.detector.YOLO")
    def test_batch_detection_flow(self, mock_yolo):
        mock_instance = MagicMock()
        mock_yolo.return_value = mock_instance
        mock_instance.names = {0: "person", 32: "sports_ball"}

        # Simulate 2 frames in batch
        mock_boxes_1 = MagicMock()
        mock_boxes_1.__len__.return_value = 1
        mock_boxes_1.xyxy.cpu().numpy.return_value = np.array([[10, 20, 30, 40]], dtype=np.float32)
        mock_boxes_1.conf.cpu().numpy.return_value = np.array([0.90], dtype=np.float32)
        mock_boxes_1.cls.cpu().numpy.return_value = np.array([0], dtype=np.float32)

        mock_boxes_2 = MagicMock()
        mock_boxes_2.__len__.return_value = 1
        mock_boxes_2.xyxy.cpu().numpy.return_value = np.array([[50, 60, 70, 80]], dtype=np.float32)
        mock_boxes_2.conf.cpu().numpy.return_value = np.array([0.88], dtype=np.float32)
        mock_boxes_2.cls.cpu().numpy.return_value = np.array([32], dtype=np.float32)

        mock_pred_1 = MagicMock()
        mock_pred_1.boxes = mock_boxes_1
        mock_pred_2 = MagicMock()
        mock_pred_2.boxes = mock_boxes_2

        mock_instance.predict.return_value = [mock_pred_1, mock_pred_2]

        detector = PlayerDetector(model_name="yolov8n.pt", device="cpu", warmup_iterations=0)
        batch = [np.zeros((720, 1280, 3), dtype=np.uint8), np.zeros((720, 1280, 3), dtype=np.uint8)]
        results = detector.detect_batch(batch, start_idx=100)

        assert len(results) == 2
        assert results[0].frame_idx == 100
        assert results[1].frame_idx == 101
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        assert results[0].class_ids[0] == 0
        assert results[1].class_ids[0] == 32
