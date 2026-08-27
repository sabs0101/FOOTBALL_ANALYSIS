"""
Optimized High-Speed TSV Football Dataset Generator.
Pre-caches AI pipeline predictions on source video frames, then writes 10,000 augmented
frame images and full Tab-Separated Values (TSV) metadata in parallel.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from tqdm import tqdm

from src.analytics.speed_distance import SpeedEstimator
from src.calibration.homography import PitchHomography
from src.detection.detector import PlayerDetector
from src.pitch.detector import PitchDetector
from src.tactics.spatial import SpatialControl
from src.team.classifier import TeamClassifier
from src.tracking.tracker import PlayerTracker


def save_single_image(args):
    img_path, frame, encode_params = args
    cv2.imwrite(str(img_path), frame, encode_params)


def generate_dataset(
    video_path: str = "data/videos/sample_broadcast.mp4",
    output_dir: str = "data/tsv_dataset",
    target_frames: int = 10000,
    device: str = "0",
    batch_img_quality: int = 80,
    max_workers: int = 8,
):
    out_path = Path(output_dir)
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    tsv_file = out_path / "dataset.tsv"

    print("=" * 65)
    print(" AI Football Analytics - High-Speed 10,000 Frame TSV Generator")
    print("=" * 65)
    print(f" Source Video     : {video_path}")
    print(f" Output Directory : {output_dir}")
    print(f" Target Frames    : {target_frames:,}")
    print(f" Output TSV File  : {tsv_file.name}")
    print(f" Compute Device   : CUDA:{device}")
    print("=" * 65)

    # Initialize AI Pipeline Modules
    detector = PlayerDetector(model_name="models/yolov8m.pt", device=device, conf_threshold=0.18, imgsz=1280)
    tracker = PlayerTracker()
    pitch_detector = PitchDetector()
    homography = PitchHomography()
    speed_estimator = SpeedEstimator()
    team_classifier = TeamClassifier()
    spatial_control = SpatialControl()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    H_pitch = homography.estimate_broadcast_homography((h, w))

    # Phase 1: Read & Run AI Pipeline on source video frames
    print(f"\n[Phase 1/3] Running AI Pipeline on {total_video_frames} source broadcast frames...")
    cached_data = []

    for f_idx in tqdm(range(total_video_frames), desc="Inference Caching"):
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, frame_idx=f_idx)
        pitch_res = pitch_detector.detect_lines(frame)
        filtered_dets = pitch_detector.filter_detections_on_pitch(detections, pitch_res.mask)
        tracked = tracker.update(filtered_dets)

        players = tracked.get_players()
        ball = tracked.get_ball()

        tids = getattr(tracked, "tracker_ids", None)
        feet_coords = tracked.get_foot_positions()
        pos_m = homography.image_to_pitch(feet_coords, H_pitch)

        speed_dict = {}
        if tids is not None and len(pos_m) > 0:
            metrics = speed_estimator.update(tids, pos_m, frame_idx=f_idx)
            for tid, pm in metrics.items():
                if pm.current_speed_kmh >= 2.0:
                    speed_dict[int(tid)] = round(pm.current_speed_kmh, 1)

        team_res = team_classifier.classify_frame(frame, tracked, pos_m)
        spatial_res = spatial_control.analyze_frame(pos_m, team_res.team_ids)

        cached_data.append({
            "raw_frame": frame,
            "num_players": len(players.xyxy),
            "has_ball": len(ball.xyxy) > 0,
            "team_a_count": team_res.team_counts.get("Team A", 0),
            "team_b_count": team_res.team_counts.get("Team B", 0),
            "top_speed": max(speed_dict.values()) if speed_dict else 0.0,
            "team_a_control_pct": spatial_res.team_a_control_pct,
            "team_b_control_pct": spatial_res.team_b_control_pct,
            "team_a_compactness_m2": spatial_res.team_a_area_m2,
            "boxes_json": json.dumps([list(map(int, b)) for b in tracked.xyxy]),
            "roles_json": json.dumps(team_res.team_names),
            "speeds_json": json.dumps(speed_dict),
        })

    cap.release()
    print(f"[Phase 1 Complete] Cached {len(cached_data)} AI frames.")

    # Phase 2: Generate 10,000 augmented frame images & TSV records
    print(f"\n[Phase 2/2] Generating {target_frames:,} augmented frames & TSV records...")
    tsv_records = []
    tsv_headers = [
        "image_name",
        "frame_idx",
        "timestamp_s",
        "num_players",
        "ball_detected",
        "team_a_count",
        "team_b_count",
        "top_speed_kmh",
        "team_a_space_dominance_pct",
        "team_b_space_dominance_pct",
        "team_a_compactness_m2",
        "bounding_boxes_json",
        "player_roles_json",
        "player_speeds_json",
    ]

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), batch_img_quality]

    # Open TSV file for streaming writes
    with open(tsv_file, "w", newline="", encoding="utf-8") as f_tsv:
        writer = csv.writer(f_tsv, delimiter="\t")
        writer.writerow(tsv_headers)

        for i in tqdm(range(target_frames), desc="Exporting TSV Dataset"):
            base_item = cached_data[i % len(cached_data)]
            aug_idx = i // len(cached_data)

            frame_aug = base_item["raw_frame"].copy()
            if aug_idx == 1:
                frame_aug = cv2.convertScaleAbs(frame_aug, alpha=1.04, beta=8)
            elif aug_idx == 2:
                frame_aug = cv2.convertScaleAbs(frame_aug, alpha=0.96, beta=-8)
            elif aug_idx == 3:
                frame_aug[:, :, 0] = np.clip(frame_aug[:, :, 0] * 0.97, 0, 255).astype(np.uint8)
                frame_aug[:, :, 2] = np.clip(frame_aug[:, :, 2] * 1.03, 0, 255).astype(np.uint8)
            elif aug_idx > 3:
                hsv = cv2.cvtColor(frame_aug, cv2.COLOR_BGR2HSV)
                scale = float(1.0 + (aug_idx % 3) * 0.04)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * scale, 0, 255).astype(np.uint8)
                frame_aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            img_name = f"frame_{i + 1:06d}.jpg"
            img_path = img_dir / img_name

            # Direct disk write (0 RAM bloat)
            cv2.imwrite(str(img_path), frame_aug, encode_params)

            timestamp = round(i / fps, 3)
            row = [
                img_name,
                str(i + 1),
                f"{timestamp:.3f}",
                str(base_item["num_players"]),
                "True" if base_item["has_ball"] else "False",
                str(base_item["team_a_count"]),
                str(base_item["team_b_count"]),
                f"{base_item['top_speed']:.1f}",
                f"{base_item['team_a_control_pct']:.1f}",
                f"{base_item['team_b_control_pct']:.1f}",
                f"{base_item['team_a_compactness_m2']:.1f}",
                base_item["boxes_json"],
                base_item["roles_json"],
                base_item["speeds_json"],
            ]
            writer.writerow(row)

    # Write summary JSON
    summary = {
        "dataset_name": "Football Tactical Computer Vision & Kinematics Dataset",
        "format": "TSV (Tab-Separated Values) + JPEG Image Frames",
        "total_images": target_frames,
        "resolution": f"{w}x{h}",
        "tsv_file": str(tsv_file.name),
        "total_tsv_rows": len(tsv_records),
        "columns": tsv_headers,
        "classes": ["Team A Outfield", "Team B Outfield", "Goalkeeper [A-GK]", "Goalkeeper [B-GK]", "Referee [REF]", "Coach [COACH]", "Football [Ball]"],
    }
    with open(out_path / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("=" * 65)
    print(" TSV DATASET GENERATION COMPLETE!")
    print("=" * 65)
    print(f" Total Frames Exported : {target_frames:,} images in {img_dir.resolve()}")
    print(f" TSV Annotations File  : {tsv_file.resolve()} ({tsv_file.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f" Dataset Summary JSON  : {(out_path / 'dataset_summary.json').resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 10,000+ Frame TSV Dataset")
    parser.add_argument("--source", type=str, default="data/videos/sample_broadcast.mp4", help="Source match video")
    parser.add_argument("--output", type=str, default="data/tsv_dataset", help="Output dataset directory")
    parser.add_argument("--frames", type=int, default=10000, help="Total target frame count")
    parser.add_argument("--device", type=str, default="0", help="GPU device index or cpu")

    args = parser.parse_args()
    generate_dataset(
        video_path=args.source,
        output_dir=args.output,
        target_frames=args.frames,
        device=args.device,
    )
