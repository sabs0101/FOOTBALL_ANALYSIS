"""
Unit tests for Quantitative Evaluation Metrics and Pipeline Evaluator.
"""

import numpy as np
import pytest
from src.evaluation.metrics import (
    compute_box_iou,
    compute_detection_metrics,
    compute_tracking_metrics,
    compute_homography_metrics,
    DetectionMetrics,
    TrackingMetrics,
    HomographyMetrics,
)


def test_box_iou_computation():
    """Verify standard IoU calculations."""
    box_a = np.array([100, 100, 200, 200], dtype=np.float32)
    box_b = np.array([100, 100, 200, 200], dtype=np.float32)
    assert compute_box_iou(box_a, box_b) == 1.0

    box_c = np.array([300, 300, 400, 400], dtype=np.float32)
    assert compute_box_iou(box_a, box_c) == 0.0

    box_d = np.array([150, 100, 250, 200], dtype=np.float32)
    iou = compute_box_iou(box_a, box_d)
    assert 0.30 <= iou <= 0.40


def test_detection_metrics_calculation():
    """Verify precision, recall, and mAP on synthetic bounding boxes."""
    pred_boxes = [
        np.array([100, 100, 200, 200]),
        np.array([300, 300, 400, 400]),
        np.array([500, 500, 600, 600]),  # False positive
    ]
    gt_boxes = [
        np.array([102, 98, 201, 202]),
        np.array([298, 305, 399, 401]),
        np.array([700, 700, 800, 800]),  # False negative
    ]

    metrics = compute_detection_metrics(pred_boxes, gt_boxes, iou_threshold=0.50)
    assert isinstance(metrics, DetectionMetrics)
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert pytest.approx(metrics.precision, 0.01) == 0.6667
    assert pytest.approx(metrics.recall, 0.01) == 0.6667


def test_homography_reprojection_rmse():
    """Verify reprojection RMSE on known point pairs."""
    pts_real = np.array([[10.0, 20.0], [50.0, 34.0], [90.0, 50.0]], dtype=np.float32)
    pts_proj = pts_real + np.array([[0.1, 0.0], [0.0, -0.1], [0.05, 0.05]])

    metrics = compute_homography_metrics(pts_proj, pts_real)
    assert isinstance(metrics, HomographyMetrics)
    assert metrics.reprojection_rmse_m < 0.20  # Less than 20cm error
    assert metrics.pitch_lock_rate_pct == 100.0
