"""
AI-Based Football Broadcast Analysis and Tactical Reconstruction
Pipeline Runner: Detection + Tracking + Pitch + Homography + Speed + Team ID + Tactics (Goalkeeper & Coach Refined)
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from src.preprocessing.preprocessor import FramePreprocessor
from src.analytics.speed_distance import SpeedEstimator
from src.calibration.homography import PitchHomography
from src.calibration.template import PitchTemplate
from src.detection.detector import PlayerDetector
from src.pitch.detector import PitchDetector
from src.tactics.heatmaps import HeatmapGenerator
from src.tactics.spatial import SpatialControl, TacticalSpatialResult
from src.team.classifier import TeamClassifier
from src.tracking.tracker import PlayerTracker
from src.utils.config import get_device, load_config
from src.utils.video import VideoReader, VideoWriter, get_video_properties
from src.visualization.annotator import VideoAnnotator
from src.visualization.radar import TacticalRadar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Football Analytics - Tactical Broadcast Reconstruction"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable CLAHE contrast enhancement and preprocessing",
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
        help="Path to save output annotated video",
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
        help="YOLO model checkpoint name (e.g. yolov8m.pt, yolov8x.pt, yolo11m.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold for player/ball detection",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image resolution (default: 1280 for sharp broadcast players)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('auto', 'cuda', 'cpu', '0')",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable multi-object tracking (detection-only mode)",
    )
    parser.add_argument(
        "--no-pitch-filter",
        action="store_true",
        help="Disable pitch mask crowd/coach detection filtering",
    )
    parser.add_argument(
        "--no-team",
        action="store_true",
        help="Disable jersey color team classification",
    )
    parser.add_argument(
        "--no-radar",
        action="store_true",
        help="Disable 2D tactical radar minimap overlay",
    )
    parser.add_argument(
        "--no-speed",
        action="store_true",
        help="Disable metric speed (km/h) calculation and badges",
    )
    parser.add_argument(
        "--no-tactics",
        action="store_true",
        help="Disable tactical spatial control and compactness analysis",
    )
    parser.add_argument(
        "--draw-pitch-lines",
        action="store_true",
        help="Highlight detected pitch field markings in the video",
    )
    parser.add_argument(
        "--draw-pitch-boundary",
        action="store_true",
        help="Highlight detected pitch polygon boundary in the video",
    )
    parser.add_argument(
        "--no-trails",
        action="store_true",
        help="Disable movement breadcrumb trails in output video",
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
    imgsz: Optional[int] = None,
    device_name: Optional[str] = None,
    enable_tracking: bool = True,
    enable_pitch_filter: bool = True,
    enable_team: bool = True,
    enable_radar: bool = True,
    enable_speed: bool = True,
    enable_tactics: bool = True,
    draw_pitch_lines: bool = False,
    draw_pitch_boundary: bool = False,
    draw_trails: bool = True,
    draw_hud: bool = True,
) -> Dict[str, Any]:
    """
    Execute end-to-end tactical analysis with refined Goalkeeper, Referee, and Coach roles.
    """
    config = load_config(config_path)

    model_name = model_name or config["detection"]["model_name"]
    conf_threshold = conf_threshold or config["detection"]["conf_threshold"]
    imgsz = imgsz or config["detection"].get("imgsz", 1280)
    iou_threshold = config["detection"]["iou_threshold"]
    preferred_device = device_name or config["system"]["device"]
    half_precision = config["system"]["half_precision"]
    model_dir = config["detection"]["model_dir"]
    filter_classes = config["detection"]["filter_classes"]

    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Input video does not exist: {source.resolve()}")

    if output_path is None:
        out_dir = Path(config["paths"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{source.stem}_tactical_master.mp4")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    props = get_video_properties(str(source))
    print(f"\n=======================================================")
    print(f" AI Football Analysis - Tactical Pipeline (GK & Refined Roles)")
    print(f"=======================================================")
    print(f" Source Video       : {source.name}")
    print(f" Resolution         : {props['width']}x{props['height']}")
    print(f" Total Frames       : {props['frame_count']}")
    print(f" Original FPS       : {props['fps']:.2f}")
    print(f" Output Video       : {output_path}")
    print(f" Model              : {model_name} (conf={conf_threshold}, imgsz={imgsz})")
    print(f" Team Classification: {'Enabled (GK & Coach Refined)' if enable_team else 'Disabled'}")
    print(f" Tactical Space/Hull: {'Enabled (Voronoi & Convex Hulls)' if enable_tactics else 'Disabled'}")
    print(f" Speed Kinematics   : {'Enabled (Regression Velocity)' if enable_speed else 'Disabled'}")
    print(f" 2D Tactical Radar  : {'Enabled (Top-Down Minimap)' if enable_radar else 'Disabled'}")
    print(f"=======================================================\n")

    # 0. Preprocessor
    prep_cfg = config.get("preprocessing", {})
    enable_preprocess = prep_cfg.get("enable_clahe", True)
    preprocessor = FramePreprocessor(
        enable_clahe=prep_cfg.get("enable_clahe", True),
        clahe_clip_limit=prep_cfg.get("clahe_clip_limit", 2.0),
        enable_gamma=prep_cfg.get("enable_gamma", False),
        gamma=prep_cfg.get("gamma", 1.15),
        denoise_method=prep_cfg.get("denoise_method", "none"),
        blur_threshold=prep_cfg.get("blur_threshold", 100.0),
    ) if enable_preprocess else None

    # 1. Detector
    detector = PlayerDetector(
        model_name=model_name,
        model_dir=model_dir,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        device=preferred_device,
        half=half_precision,
        filter_classes=filter_classes,
    )

    # 2. Pitch Detector
    pitch_cfg = config.get("pitch", {})
    pitch_detector = PitchDetector(
        hsv_green_lower=tuple(pitch_cfg.get("hsv_green_lower", [32, 45, 40])),
        hsv_green_upper=tuple(pitch_cfg.get("hsv_green_upper", [85, 255, 235])),
        morph_kernel_size=pitch_cfg.get("morph_kernel_size", 15),
        min_pitch_area_ratio=pitch_cfg.get("min_pitch_area_ratio", 0.20),
        white_contrast_thresh=pitch_cfg.get("white_contrast_thresh", 18),
        min_line_length=pitch_cfg.get("min_line_length", 60),
        max_line_gap=pitch_cfg.get("max_line_gap", 40),
    )

    # 3. Tracker
    tracker = None
    if enable_tracking:
        track_cfg = config.get("tracking", {})
        tracker = PlayerTracker(
            track_activation_threshold=track_cfg.get("track_activation_threshold", 0.20),
            lost_track_buffer=track_cfg.get("lost_track_buffer", 30),
            minimum_matching_threshold=track_cfg.get("minimum_matching_threshold", 0.70),
            frame_rate=int(props["fps"]),
            trail_length=track_cfg.get("trail_length", 20),
        )

    # 4. Team Classifier
    team_classifier = None
    if enable_team:
        team_cfg = config.get("team", {})
        team_classifier = TeamClassifier(
            n_teams=team_cfg.get("n_teams", 2),
            history_window=team_cfg.get("history_window", 20),
            referee_dist_threshold=team_cfg.get("referee_dist_threshold", 85.0),
        )

    # 5. Homography & 2D Radar
    calibrator = PitchHomography()
    radar_cfg = config.get("radar", {})
    tactical_radar = None
    if enable_radar:
        tactical_radar = TacticalRadar(
            radar_width=radar_cfg.get("radar_width", 380),
            radar_height=radar_cfg.get("radar_height", 245),
            player_dot_radius=radar_cfg.get("player_dot_radius", 6),
            ball_dot_radius=radar_cfg.get("ball_dot_radius", 5),
        )

    # 6. Speed Estimator
    speed_estimator = None
    if enable_speed:
        analytics_cfg = config.get("analytics", {})
        speed_estimator = SpeedEstimator(
            fps=props["fps"],
            window_size=analytics_cfg.get("window_size", 18),
            max_speed_kmh=analytics_cfg.get("max_realistic_speed_kmh", 38.0),
            min_speed_kmh=analytics_cfg.get("min_speed_threshold_kmh", 2.5),
        )

    # 7. Tactical Spatial Control & Heatmaps
    spatial_control = None
    heatmap_gen = None
    if enable_tactics:
        spatial_control = SpatialControl()
        heatmap_gen = HeatmapGenerator()

    # 8. Annotator
    annotator = VideoAnnotator(
        box_thickness=config["visualization"]["box_thickness"],
        font_scale=config["visualization"]["text_scale"],
        draw_conf=config["visualization"]["draw_confidences"],
        draw_hud=draw_hud and config["visualization"]["draw_hud"],
        draw_tracks=enable_tracking and config["visualization"]["draw_tracks"],
        draw_trails=draw_trails and config["visualization"]["draw_trails"],
        draw_speed=enable_speed and config["visualization"].get("draw_speed", True),
        draw_team=enable_team and config["visualization"].get("draw_team", True),
        draw_pitch_boundary=draw_pitch_boundary or config["visualization"].get("draw_pitch_boundary", False),
        draw_pitch_lines=draw_pitch_lines or config["visualization"].get("draw_pitch_lines", False),
    )

    # Video IO
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
    unique_track_ids = set()
    ball_detected_count = 0
    pitch_detected_count = 0
    avg_team_a_control = []
    avg_team_b_control = []

    print(f"[Processing] Running Master Tactical Pipeline across {total_frames} frames...")

    try:
        for frame_idx, frame in tqdm(reader, total=total_frames, desc="Tactical Master Pipeline"):
            t0 = time.time()

            # Step 0: Preprocessing & Contrast Enhancement
            proc_frame = preprocessor.process(frame).frame if preprocessor is not None else frame

            # Step 1: Pitch & Line Detection
            pitch_result = pitch_detector.detect_lines(proc_frame)
            if pitch_result.pitch_area_ratio >= 0.20:
                pitch_detected_count += 1

            # Step 2: YOLO Detection (imgsz=1280)
            detections = detector.detect(proc_frame, frame_idx=frame_idx)

            # Step 3: Touchline-bounded Crowd & Dugout Filtering
            if enable_pitch_filter and pitch_result.pitch_area_ratio >= 0.20:
                detections = pitch_detector.filter_detections_on_pitch(detections, pitch_result.mask)

            # Step 4: Multi-Object Tracking (ByteTrack)
            if tracker is not None:
                processed_results = tracker.update(detections)
                for tid in processed_results.tracker_ids:
                    if tid >= 0:
                        unique_track_ids.add(int(tid))
            else:
                processed_results = detections

            players_res = processed_results.get_players()
            num_players = len(players_res.xyxy)
            player_counts.append(num_players)

            ball_res = processed_results.get_ball()
            has_ball = len(ball_res.xyxy) > 0
            if has_ball:
                ball_detected_count += 1

            # Step 5: Homography & Metric Coordinates Calculation
            player_positions_m = np.empty((0, 2), dtype=np.float32)
            ball_pos_m = None

            if pitch_result.pitch_area_ratio >= 0.20 and num_players > 0:
                homography_res = calibrator.estimate_broadcast_homography(
                    frame_shape=(props["height"], props["width"]),
                    top_touchline_y=245.0,
                    bottom_touchline_y=745.0,
                    halfway_x=960.0,
                    center_y=500.0,
                )

                if homography_res.is_valid:
                    feet_pts = players_res.get_foot_positions()
                    player_positions_m = calibrator.image_to_pitch(feet_pts, homography_res.H)

                    if has_ball:
                        ball_box = ball_res.xyxy[0]
                        ball_center = np.array([[(ball_box[0] + ball_box[2]) / 2.0, (ball_box[1] + ball_box[3]) / 2.0]], dtype=np.float32)
                        ball_m = calibrator.image_to_pitch(ball_center, homography_res.H)
                        if len(ball_m) > 0:
                            ball_pos_m = (float(ball_m[0, 0]), float(ball_m[0, 1]))

            # Step 6: Team, Goalkeeper, Referee, and Coach Classification
            team_result = None
            if team_classifier is not None:
                team_result = team_classifier.classify_frame(
                    frame=frame,
                    detections=processed_results,
                    positions_m=player_positions_m if len(player_positions_m) == len(processed_results.xyxy) else None,
                )

            # Step 7: Speed & Distance Estimation
            player_metrics = {}
            team_summary = {}
            if speed_estimator is not None and len(player_positions_m) > 0:
                player_metrics = speed_estimator.update(
                    track_ids=players_res.tracker_ids,
                    positions_m=player_positions_m,
                    frame_idx=frame_idx,
                )
                team_summary = speed_estimator.get_team_summary()

            # Step 8: Tactical Spatial Control & Convex Hulls
            tactical_spatial_result = None
            if spatial_control is not None and team_result is not None and len(player_positions_m) > 0:
                player_indices = np.where(processed_results.class_ids == 0)[0]
                player_team_ids = team_result.team_ids[player_indices] if len(player_indices) > 0 else np.empty((0,), dtype=int)
                
                tactical_spatial_result = spatial_control.analyze_frame(
                    positions_m=player_positions_m,
                    team_ids=player_team_ids,
                )
                avg_team_a_control.append(tactical_spatial_result.team_a_control_pct)
                avg_team_b_control.append(tactical_spatial_result.team_b_control_pct)

                # Accumulate heatmap density points
                if heatmap_gen is not None:
                    heatmap_gen.add_positions(
                        track_ids=players_res.tracker_ids,
                        positions_m=player_positions_m,
                        team_ids=player_team_ids,
                    )

            dt = time.time() - t0
            frame_times.append(dt)
            fps_inst = 1.0 / dt if dt > 0 else 0.0

            # Step 9: Visual Annotation & Telemetry HUD
            annotated_frame = annotator.annotate(
                frame=frame,
                detections=processed_results,
                pitch_result=pitch_result,
                team_result=team_result,
                tactical_spatial_result=tactical_spatial_result,
                player_metrics=player_metrics,
                team_summary=team_summary,
                fps=fps_inst,
                frame_idx=frame_idx + 1,
                total_frames=total_frames,
                device_name=device_label,
            )

            # Step 10: 2D Tactical Radar Minimap Overlay
            if tactical_radar is not None and len(player_positions_m) > 0:
                player_team_colors = None
                if team_result is not None:
                    player_indices = np.where(processed_results.class_ids == 0)[0]
                    player_team_colors = [team_result.team_colors[idx] for idx in player_indices if idx < len(team_result.team_colors)]

                radar_img = tactical_radar.render_radar(
                    player_positions_m=player_positions_m,
                    player_track_ids=players_res.tracker_ids,
                    ball_position_m=ball_pos_m,
                    team_colors=player_team_colors,
                    tactical_spatial_result=tactical_spatial_result,
                )
                annotated_frame = tactical_radar.overlay_on_frame(
                    annotated_frame,
                    radar_img,
                    position=radar_cfg.get("position", "bottom_right"),
                    alpha=radar_cfg.get("alpha", 0.95),
                )

            writer.write(annotated_frame)

    finally:
        reader.release()
        writer.release()

    # Step 11: Export Tactical Positional Heatmaps
    if heatmap_gen is not None:
        heatmap_dir = config.get("tactics", {}).get("heatmap_output_dir", "outputs/heatmaps")
        heatmap_gen.export_all_heatmaps(heatmap_dir)

    total_time = time.time() - start_time
    avg_fps = len(frame_times) / total_time if total_time > 0 else 0.0
    avg_players = float(np.mean(player_counts)) if player_counts else 0.0
    final_team_summary = speed_estimator.get_team_summary() if speed_estimator else {}
    mean_ctrl_a = float(np.mean(avg_team_a_control)) if avg_team_a_control else 50.0
    mean_ctrl_b = float(np.mean(avg_team_b_control)) if avg_team_b_control else 50.0

    print(f"\n=======================================================")
    print(f" Processing Complete!")
    print(f"=======================================================")
    print(f" Total Elapsed Time   : {total_time:.2f} seconds")
    print(f" Average Speed        : {avg_fps:.2f} FPS")
    print(f" Avg On-Pitch Players : {avg_players:.1f}")
    print(f" Pitch Locked         : {pitch_detected_count}/{total_frames} frames ({pitch_detected_count/max(1,total_frames)*100:.1f}%)")
    print(f" Team A Dominance     : {mean_ctrl_a:.1f}% pitch space control")
    print(f" Team B Dominance     : {mean_ctrl_b:.1f}% pitch space control")
    if enable_tracking:
        print(f" Total Unique Tracks  : {len(unique_track_ids)} unique IDs assigned")
    print(f" Peak Sprint Speed    : {final_team_summary.get('max_speed_kmh', 0.0)} km/h")
    print(f" Total Distance Run   : {final_team_summary.get('total_distance_km', 0.0)} km")
    print(f" Ball Detected In     : {ball_detected_count}/{total_frames} frames ({ball_detected_count/max(1,total_frames)*100:.1f}%)")
    print(f" Output Video         : {output_path}")
    print(f" Heatmaps Saved To    : outputs/heatmaps/")
    print(f"=======================================================\n")

    log_dir = Path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "source": str(source),
        "output": output_path,
        "model": model_name,
        "imgsz": imgsz,
        "device": device_label,
        "team_classification_enabled": enable_team,
        "tactical_space_control_enabled": enable_tactics,
        "team_a_mean_control_pct": round(mean_ctrl_a, 2),
        "team_b_mean_control_pct": round(mean_ctrl_b, 2),
        "team_physical_summary": final_team_summary,
        "pitch_locked_frames": pitch_detected_count,
        "total_frames": total_frames,
        "elapsed_seconds": round(total_time, 2),
        "average_fps": round(avg_fps, 2),
        "average_on_pitch_players": round(avg_players, 2),
        "total_unique_tracks": len(unique_track_ids) if enable_tracking else None,
        "ball_detection_rate": round(ball_detected_count / max(1, total_frames), 4),
    }

    log_path = log_dir / f"{source.stem}_tactical_master_summary.json"
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
        imgsz=args.imgsz,
        device_name=args.device,
        enable_tracking=not args.no_track,
        enable_pitch_filter=not args.no_pitch_filter,
        enable_team=not args.no_team,
        enable_radar=not args.no_radar,
        enable_speed=not args.no_speed,
        enable_tactics=not args.no_tactics,
        draw_pitch_lines=args.draw_pitch_lines,
        draw_pitch_boundary=args.draw_pitch_boundary,
        draw_trails=not args.no_trails,
        draw_hud=not args.no_hud,
    )
