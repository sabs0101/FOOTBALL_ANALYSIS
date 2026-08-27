"""
Unit tests for Preprocessing, Contrast Enhancement, and Exploratory Data Analysis (EDA).
"""

import numpy as np
import pytest
from src.preprocessing.preprocessor import FramePreprocessor, PreprocessingResult
from src.preprocessing.eda import VideoEDA, EDAReport


def test_frame_preprocessor_clahe_enhancement():
    """Verify CLAHE contrast enhancement runs and enhances contrast without altering shape."""
    h, w = 720, 1280
    frame = np.random.randint(40, 180, size=(h, w, 3), dtype=np.uint8)

    preprocessor = FramePreprocessor(enable_clahe=True, enable_gamma=False)
    result = preprocessor.process(frame)

    assert isinstance(result, PreprocessingResult)
    assert result.frame.shape == (h, w, 3)
    assert result.avg_brightness > 0
    assert result.blur_score >= 0.0


def test_frame_preprocessor_motion_blur_detection():
    """Verify Laplacian operator detects blurry frames."""
    h, w = 480, 640
    sharp_frame = np.zeros((h, w, 3), dtype=np.uint8)
    sharp_frame[::20, :] = 255  # High-frequency sharp lines

    blurry_frame = np.full((h, w, 3), 120, dtype=np.uint8)  # Uniform flat blur

    preprocessor = FramePreprocessor(blur_threshold=50.0)

    res_sharp = preprocessor.process(sharp_frame)
    res_blurry = preprocessor.process(blurry_frame)

    assert res_sharp.blur_score > res_blurry.blur_score
    assert res_blurry.is_blurry is True


def test_eda_analysis_on_sample_clip(tmp_path):
    """Verify VideoEDA generates structured metadata and report."""
    import cv2
    video_path = tmp_path / "test_pitch.mp4"

    # Create synthetic test video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (640, 360))
    for _ in range(30):
        frame = np.full((360, 640, 3), (35, 140, 35), dtype=np.uint8)  # Green field
        out.write(frame)
    out.release()

    eda = VideoEDA()
    report = eda.analyze_video(str(video_path), sample_step=5, max_samples=10)

    assert isinstance(report, EDAReport)
    assert report.metadata.width == 640
    assert report.metadata.height == 360
    assert report.metadata.total_frames == 30
    assert report.grass_coverage_pct > 80.0
    assert len(report.recommendations) > 0
