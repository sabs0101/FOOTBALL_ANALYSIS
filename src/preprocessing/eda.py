"""
Exploratory Data Analysis (EDA) Module for Football Video Streams.
Extracts video diagnostics, turf color distributions, lighting variance,
and frame-to-frame motion energy.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Basic stream metadata."""
    file_name: str
    width: int
    height: int
    total_frames: int
    fps: float
    duration_seconds: float
    aspect_ratio: str


@dataclass
class EDAReport:
    """Comprehensive Exploratory Data Analysis report for match footage."""
    metadata: VideoMetadata
    mean_brightness: float
    std_brightness: float
    mean_blur_score: float
    blurry_frames_pct: float
    grass_coverage_pct: float
    dominant_turf_hsv: List[float]
    sample_frames_analyzed: int
    recommendations: List[str]


class VideoEDA:
    """
    Automated Exploratory Data Analysis engine for football match broadcasts.
    """

    def __init__(
        self,
        hsv_green_lower: np.ndarray = np.array([32, 45, 40], dtype=np.uint8),
        hsv_green_upper: np.ndarray = np.array([85, 255, 235], dtype=np.uint8),
    ):
        self.green_lower = hsv_green_lower
        self.green_upper = hsv_green_upper

    def analyze_video(
        self,
        video_path: str,
        sample_step: int = 25,
        max_samples: int = 50,
    ) -> EDAReport:
        """
        Analyze video stream across uniform sampled intervals.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0.0

        gcd = np.gcd(w, h)
        aspect = f"{w // gcd}:{h // gcd}"

        meta = VideoMetadata(
            file_name=path.name,
            width=w,
            height=h,
            total_frames=total_frames,
            fps=round(fps, 2),
            duration_seconds=round(duration, 2),
            aspect_ratio=aspect,
        )

        brightness_list = []
        blur_list = []
        grass_cover_list = []
        turf_hsv_list = []

        step = max(1, total_frames // max_samples) if total_frames > max_samples else sample_step
        frame_idx = 0
        analyzed_count = 0

        while frame_idx < total_frames and analyzed_count < max_samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Turf coverage analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            grass_pct = float(np.sum(mask > 0) / (w * h) * 100.0)

            turf_pixels = hsv[mask > 0]
            if len(turf_pixels) > 0:
                mean_turf = np.mean(turf_pixels, axis=0)
                turf_hsv_list.append(mean_turf)

            brightness_list.append(brightness)
            blur_list.append(blur)
            grass_cover_list.append(grass_pct)

            analyzed_count += 1
            frame_idx += step

        cap.release()

        mean_bright = float(np.mean(brightness_list)) if brightness_list else 0.0
        std_bright = float(np.std(brightness_list)) if brightness_list else 0.0
        mean_blur = float(np.mean(blur_list)) if blur_list else 0.0
        blurry_pct = float(np.sum(np.array(blur_list) < 100.0) / len(blur_list) * 100.0) if blur_list else 0.0
        mean_grass = float(np.mean(grass_cover_list)) if grass_cover_list else 0.0
        dom_turf = list(map(float, np.mean(turf_hsv_list, axis=0))) if turf_hsv_list else [55.0, 150.0, 120.0]

        # Generate automated engineering recommendations
        recs = []
        if mean_bright < 80.0:
            recs.append("Video has low illumination (<80). Enable CLAHE or Gamma correction in preprocessor.")
        elif mean_bright > 180.0:
            recs.append("Video has bright exposure. Consider histogram normalization.")

        if blurry_pct > 15.0:
            recs.append(f"Significant motion blur detected in {blurry_pct:.1f}% of frames. Enable Kalman coasting.")

        if mean_grass > 50.0:
            recs.append(f"Pitch grass dominates {mean_grass:.1f}% of view. Ideal for HSV color segmentation.")
        else:
            recs.append("Low pitch visibility (<50%). Camera might be zoomed in on player close-ups.")

        return EDAReport(
            metadata=meta,
            mean_brightness=round(mean_bright, 2),
            std_brightness=round(std_bright, 2),
            mean_blur_score=round(mean_blur, 2),
            blurry_frames_pct=round(blurry_pct, 2),
            grass_coverage_pct=round(mean_grass, 2),
            dominant_turf_hsv=[round(x, 1) for x in dom_turf],
            sample_frames_analyzed=analyzed_count,
            recommendations=recs,
        )

    def save_report(self, report: EDAReport, output_path: str):
        """Save EDA report as formatted JSON and Markdown."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(report)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        # Also write Markdown summary
        md_path = out.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Exploratory Data Analysis (EDA) Report\n\n")
            f.write(f"**Target File**: `{report.metadata.file_name}`\n")
            f.write(f"- **Resolution**: {report.metadata.width}x{report.metadata.height} ({report.metadata.aspect_ratio})\n")
            f.write(f"- **Total Frames / Duration**: {report.metadata.total_frames} frames ({report.metadata.duration_seconds}s @ {report.metadata.fps} FPS)\n")
            f.write(f"- **Mean Brightness**: {report.mean_brightness:.1f} / 255 (Std: {report.std_brightness:.1f})\n")
            f.write(f"- **Mean Sharpness (Laplacian Var)**: {report.mean_blur_score:.1f}\n")
            f.write(f"- **Blurry Frames**: {report.blurry_frames_pct:.1f}%\n")
            f.write(f"- **Pitch Grass Coverage**: {report.grass_coverage_pct:.1f}%\n")
            f.write(f"- **Dominant Turf HSV**: Hue={report.dominant_turf_hsv[0]}, Sat={report.dominant_turf_hsv[1]}, Val={report.dominant_turf_hsv[2]}\n\n")
            f.write("### Pipeline Recommendations:\n")
            for r in report.recommendations:
                f.write(f"- {r}\n")

        print(f"[EDA] Successfully generated EDA report: {out.resolve()} and {md_path.resolve()}")
