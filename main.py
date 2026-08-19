"""
AI-Based Football Broadcast Analysis and Tactical Reconstruction
Milestone 1: Player Detection & Video Annotation Pipeline

Entry point for video inference.
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional
import cv2
import numpy as np
from tqdm import tqdm

from src.detection.detector import PlayerDetector
from src.utils.config import get_device, load_config
from src.utils.video import VideoReader, VideoWriter, get_video_properties
from src.visualization.annotator import VideoAnnotator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Football Analytics - Milestone 1: Player & Ball Detection"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/videos/sample_broadcast.mp4",
        help="Path to input video file or image directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output annotated video (defaults to outputs/detections/<name>_detected.mp4)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLO model checkpoint name (e.g. yolov8m.pt, yolov8x.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold for player/ball detection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('auto', 'cuda', 'cpu', '0')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable HUD overlay in output video",
    )
    return parser.parse_args()


def run_pipeline(
    source_path: str,
    output_path: Optional[str] = None,
    config_path: str = "config.yaml",
    model_name: Optional[str] = None,
    conf_threshold: Optional[float] = None,
    device_name: Optional[str] = None,
    batch_size: int = 1,
    draw_hud: bool = True,
) -> Dict[str, Any]:
    """
    Execute end-to-end player detection pipeline on input video.
    """
    config = load_config(config_path)

    # Resolve arguments with fallback to config
    model_name = model_name or config["detection"]["model_name"]
    conf_threshold = conf_threshold or config["detection"]["conf_threshold"]
    iou_threshold = config["detection"]["iou_threshold"]
    preferred_device = device_name or config["system"]["device"]
    half_precision = config["system"]["half_precision"]
    model_dir = config["detection"]["model_dir"]
    filter_classes = config["detection"]["filter_classes"]

    # Verify input video
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Input video does not exist: {source.resolve()}")

    # Setup output path
    if output_path is None:
        out_dir = Path(config["paths"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{source.stem}_detected.mp4")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Get input video properties
    props = get_video_properties(str(source))
    print(f"\n=======================================================")
    print(f" AI Football Analysis - Player Detection (Milestone 1)")
    print(f"=======================================================")
    print(f" Source Video     : {source.name}")
    print(f" Resolution       : {props['width']}x{props['height']}")
    print(f" Total Frames     : {props['frame_count']}")
    print(f" Original FPS     : {props['fps']:.2f}")
    print(f" Output Video     : {output_path}")
    print(f" Model            : {model_name} (conf={conf_threshold})")
    print(f"=======================================================\n")

    # Initialize Detector
    detector = PlayerDetector(
        model_name=model_name,
        model_dir=model_dir,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=preferred_device,
        half=half_precision,
        filter_classes=filter_classes,
    )

    # Initialize Visual Annotator
    annotator = VideoAnnotator(
        box_thickness=config["visualization"]["box_thickness"],
        font_scale=config["visualization"]["text_scale"],
        draw_conf=config["visualization"]["draw_confidences"],
        draw_hud=draw_hud and config["visualization"]["draw_hud"],
    )

    # Initialize Video IO
    reader = VideoReader(str(source))
    writer = VideoWriter(
        output_path=output_path,
        fps=props["fps"],
        width=props["width"],
        height=props["height"],
    )

    device_label = "RTX 4050 (CUDA)" if "cuda" in detector.device else "CPU"
    total_frames = props["frame_count"]
    start_time = time.time()
    frame_times = []
    player_counts = []
    ball_detected_count = 0

    print(f"[Processing] Running inference across {total_frames} frames...")

    try:
        if batch_size <= 1:
            for frame_idx, frame in tqdm(reader, total=total_frames, desc="Inferring"):
                t0 = time.time()
                detections = detector.detect(frame, frame_idx=frame_idx)
                dt = time.time() - t0
                frame_times.append(dt)

                fps_inst = 1.0 / dt if dt > 0 else 0.0
                num_players = len(detections.get_players().xyxy)
                player_counts.append(num_players)
                if len(detections.get_ball().xyxy) > 0:
                    ball_detected_count += 1

                annotated_frame = annotator.annotate(
                    frame=frame,
                    detections=detections,
                    fps=fps_inst,
                    frame_idx=frame_idx + 1,
                    total_frames=total_frames,
                    device_name=device_label,
                )
                writer.write(annotated_frame)
        else:
            # Batch processing loop
            batch_frames = []
            batch_indices = []
            for frame_idx, frame in tqdm(reader, total=total_frames, desc="Inferring (Batched)"):
                batch_frames.append(frame)
                batch_indices.append(frame_idx)

                if len(batch_frames) == batch_size or frame_idx == total_frames - 1:
                    t0 = time.time()
                    batch_detections = detector.detect_batch(
                        batch_frames, start_idx=batch_indices[0]
                    )
                    dt = (time.time() - t0) / len(batch_frames)

                    for b_frame, b_idx, b_det in zip(
                        batch_frames, batch_indices, batch_detections
                    ):
                        frame_times.append(dt)
                        fps_inst = 1.0 / dt if dt > 0 else 0.0
                        num_players = len(b_det.get_players().xyxy)
                        player_counts.append(num_players)
                        if len(b_det.get_ball().xyxy) > 0:
                            ball_detected_count += 1

                        annotated_frame = annotator.annotate(
                            frame=b_frame,
                            detections=b_det,
                            fps=fps_inst,
                            frame_idx=b_idx + 1,
                            total_frames=total_frames,
                            device_name=device_label,
                        )
                        writer.write(annotated_frame)

                    batch_frames.clear()
                    batch_indices.clear()

    finally:
        reader.release()
        writer.release()

    total_time = time.time() - start_time
    avg_fps = len(frame_times) / total_time if total_time > 0 else 0.0
    avg_players = float(np.mean(player_counts)) if player_counts else 0.0

    print(f"\n=======================================================")
    print(f" Processing Complete!")
    print(f"=======================================================")
    print(f" Total Elapsed Time : {total_time:.2f} seconds")
    print(f" Average Speed      : {avg_fps:.2f} FPS")
    print(f" Avg Players/Frame  : {avg_players:.1f}")
    print(f" Ball Detected In   : {ball_detected_count}/{total_frames} frames ({ball_detected_count/max(1,total_frames)*100:.1f}%)")
    print(f" Output Saved To    : {output_path}")
    print(f"=======================================================\n")

    # Save execution metrics to log
    log_dir = Path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "source": str(source),
        "output": output_path,
        "model": model_name,
        "device": device_label,
        "total_frames": total_frames,
        "elapsed_seconds": round(total_time, 2),
        "average_fps": round(avg_fps, 2),
        "average_players_per_frame": round(avg_players, 2),
        "ball_detection_rate": round(ball_detected_count / max(1, total_frames), 4),
    }

    log_path = log_dir / f"{source.stem}_detection_summary.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        source_path=args.source,
        output_path=args.output,
        config_path=args.config,
        model_name=args.model,
        conf_threshold=args.conf,
        device_name=args.device,
        batch_size=args.batch_size,
        draw_hud=not args.no_hud,
    )
