"""
Unit tests for Milestone 15: Asynchronous Multi-Threaded Producer-Consumer Architecture.
"""

from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import pytest

from src.pipeline.async_pipeline import (
    AsyncTacticalPipeline,
    FramePacket,
    TrackingPacket,
    AnalyticsPacket,
    PipelineMetrics,
)
from src.detection.detector import DetectionResult
from src.pitch.detector import PitchResult


def _create_synthetic_test_video(path: str, num_frames: int = 20, width: int = 640, height: int = 360, fps: int = 25):
    """Generates a synthetic MP4 test video with a moving soccer player and ball."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    for i in range(num_frames):
        # Grass background
        frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)
        # White touchlines
        cv2.line(frame, (50, 50), (width - 50, 50), (255, 255, 255), 2)
        cv2.line(frame, (50, height - 50), (width - 50, height - 50), (255, 255, 255), 2)
        cv2.line(frame, (width // 2, 50), (width // 2, height - 50), (255, 255, 255), 2)

        # Player 1 (Red) moving right
        x1 = int(100 + i * 5)
        y1 = 150
        cv2.rectangle(frame, (x1, y1), (x1 + 30, y1 + 60), (0, 0, 220), -1)

        # Player 2 (Blue) moving left
        x2 = int(450 - i * 4)
        y2 = 180
        cv2.rectangle(frame, (x2, y2), (x2 + 30, y2 + 60), (220, 50, 0), -1)

        # Ball (White circle) moving
        bx = int(120 + i * 8)
        by = int(170 + i * 2)
        cv2.circle(frame, (bx, by), 6, (255, 255, 255), -1)

        writer.write(frame)
    writer.release()


def test_async_pipeline_initialization():
    """Verify AsyncTacticalPipeline configuration and thread queue structure."""
    pipeline = AsyncTacticalPipeline(
        options={
            "detector_interval": 2,
            "sparse_gme": True,
            "homography_interval": 10,
            "clahe": False,
        },
        queue_size=16,
    )
    assert pipeline.detector_interval == 2
    assert pipeline.sparse_gme is True
    assert pipeline.homography_interval == 10
    assert pipeline.enable_clahe is False
    assert pipeline.queue_size == 16
    assert pipeline._ingest_queue.maxsize == 16
    assert pipeline._inference_queue.maxsize == 16
    assert pipeline._analytics_queue.maxsize == 16


def test_telemetry_dict_serialization():
    """Verify format and types in _extract_telemetry_dict for real-time live streaming."""
    pipeline = AsyncTacticalPipeline()
    # Dummy packets
    det = DetectionResult(
        xyxy=np.array([[100, 100, 130, 160], [200, 200, 210, 210]], dtype=np.float32),
        confidences=np.array([0.90, 0.85], dtype=np.float32),
        class_ids=np.array([0, 32], dtype=int),
        class_names=["person", "sports ball"],
        tracker_ids=np.array([10, -1], dtype=int),
        frame_idx=0,
    )
    pitch_res = PitchResult(
        mask=np.ones((100, 100), dtype=np.uint8) * 255,
        polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32),
        lines=np.empty((0, 4), dtype=np.int32),
        pitch_area_ratio=0.85,
    )
    packet = AnalyticsPacket(
        frame_idx=15,
        raw_frame=np.zeros((100, 100, 3), dtype=np.uint8),
        detections=det,
        pitch_result=pitch_res,
        team_result=None,
        tactical_spatial_result=None,
        player_metrics={},
        team_summary={},
        ball_state=None,
        possession_result=None,
        camera_motion=None,
        cut_result=None,
        active_event=None,
        player_positions_m=np.array([[52.5, 34.0]], dtype=np.float32),
        ball_pos_m=(50.0, 30.0),
    )

    telemetry = pipeline._extract_telemetry_dict(
        packet=packet,
        frame_idx=15,
        total_frames=100,
        fps=45.0,
        elapsed_time=0.33,
    )

    assert telemetry["frame_idx"] == 15
    assert telemetry["total_frames"] == 100
    assert telemetry["fps"] == 45.0
    assert isinstance(telemetry["players"], list)
    assert len(telemetry["players"]) == 1
    assert telemetry["players"][0]["id"] == 10
    assert telemetry["ball"]["detected"] is False
    assert telemetry["ball"]["pos_m"] == [50.0, 30.0]
    assert "tactics" in telemetry
    assert "camera" in telemetry


def test_async_pipeline_synthetic_video_execution():
    """Verify end-to-end 4-stage pipeline execution on a synthetic video."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_in = str(Path(tmp_dir) / "test_synth.mp4")
        video_out = str(Path(tmp_dir) / "test_synth_out.mp4")
        _create_synthetic_test_video(video_in, num_frames=12, width=640, height=360, fps=25)

        processed_frames_telemetry = []
        progress_updates = []

        def on_frame(f_idx, telem, frame_bgr):
            processed_frames_telemetry.append((f_idx, telem))
            assert isinstance(frame_bgr, np.ndarray)
            assert frame_bgr.shape[0] == 360
            assert frame_bgr.shape[1] == 640

        def on_prog(cur, tot, fps):
            progress_updates.append((cur, tot, fps))

        pipeline = AsyncTacticalPipeline(
            options={
                "detector_interval": 2,
                "sparse_gme": True,
                "homography_interval": 5,
                "device": "cpu",
            },
            on_frame_processed=on_frame,
            on_progress=on_prog,
        )

        results = pipeline.run(source_path=video_in, output_path=video_out)

        assert results["total_frames"] == 12
        assert results["processed_frames"] == 12
        assert results["average_fps"] > 0
        assert len(processed_frames_telemetry) == 12
        assert len(progress_updates) > 0
        assert Path(video_out).exists()
        assert Path(video_out).stat().st_size > 0
        assert "stage_latencies_ms" in results


def test_async_pipeline_graceful_stop():
    """Verify premature stop() terminates all worker threads cleanly without deadlock."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_in = str(Path(tmp_dir) / "test_long.mp4")
        _create_synthetic_test_video(video_in, num_frames=30, width=640, height=360, fps=25)

        pipeline = AsyncTacticalPipeline(options={"device": "cpu"})
        pipeline.start(source_path=video_in)

        # Allow pipeline to process first 2-3 frames then signal immediate stop
        time.sleep(0.08)
        pipeline.stop()

        # join should return without hanging
        pipeline.join(timeout=3.0)

        for t in pipeline._threads:
            assert not t.is_alive(), f"Thread {t.name} failed to terminate after stop()"
