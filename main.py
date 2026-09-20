"""
AI-Based Football Broadcast Analysis and Tactical Reconstruction
Pipeline Runner: Detection + Tracking + CMC Camera Compensation + Pitch + Homography + Speed + Team ID + Tactics + Ball Tracking & Possession
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
from src.analytics.events import EventDetector, MatchEvent, EventSummary
from src.calibration.homography import PitchHomography
from src.calibration.template import PitchTemplate
from src.calibration.camera_motion import CameraMotionCompensator, CameraMotionResult
from src.detection.detector import PlayerDetector
from src.pitch.detector import PitchDetector
from src.tactics.heatmaps import HeatmapGenerator
from src.tactics.spatial import SpatialControl, TacticalSpatialResult
from src.team.classifier import TeamClassifier
from src.tracking.tracker import PlayerTracker
from src.tracking.ball_tracker import BallTracker, BallState, PossessionResult
from src.tracking.cut_detector import CameraCutDetector, CutDetectionResult
from src.tracking.reid import PlayerReID
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
        "--no-ball-track",
        action="store_true",
        help="Disable Kalman ball tracking, interpolation, and possession assignment",
    )
    parser.add_argument(
        "--no-cmc",
        action="store_true",
        help="Disable Camera Motion Compensation (GME)",
    )
    parser.add_argument(
        "--no-cut-detect",
        action="store_true",
        help="Disable camera shot transition & cut detection",
    )
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Disable cross-cut player appearance Re-ID matching",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Disable discrete football match event recognition (passes, shots, duels)",
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
    enable_ball_tracking: bool = True,
    enable_cmc: bool = True,
    enable_cut_detection: bool = True,
    enable_reid: bool = True,
    enable_events: bool = True,
    draw_pitch_lines: bool = False,
    draw_pitch_boundary: bool = False,
    draw_trails: bool = True,
    draw_hud: bool = True,
) -> Dict[str, Any]:
    """
    Execute end-to-end tactical analysis with Camera Motion Compensation, Camera Cut Detection,
    Player Re-Identification, Discrete Football Match Event Recognition, Kalman Ball Tracking, and Possession Assignment.
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
    print(f" AI Football Analysis - Tactical Pipeline (Milestone 10)")
    print(f"=======================================================")
    print(f" Source Video       : {source.name}")
    print(f" Resolution         : {props['width']}x{props['height']}")
    print(f" Total Frames       : {props['frame_count']}")
    print(f" Original FPS       : {props['fps']:.2f}")
    print(f" Output Video       : {output_path}")
    print(f" Model              : {model_name} (conf={conf_threshold}, imgsz={imgsz})")
    print(f" Camera Motion (CMC): {'Enabled (GME & PTZ Telemetry)' if enable_cmc else 'Disabled'}")
    print(f" Cut Detection/Re-ID: {'Enabled (HSV/Edge Cut + Hungarian Re-ID)' if (enable_cut_detection and enable_reid) else 'Disabled'}")
    print(f" Team Classification: {'Enabled (GK & Coach Refined)' if enable_team else 'Disabled'}")
    print(f" Tactical Space/Hull: {'Enabled (Voronoi & Convex Hulls)' if enable_tactics else 'Disabled'}")
    print(f" Ball & Possession  : {'Enabled (Kalman Smoothing & Possession)' if enable_ball_tracking else 'Disabled'}")
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

    # 2. Camera Motion Compensator & Cut/Re-ID Engines (Milestones 9 & 10)
    camera_compensator = CameraMotionCompensator() if enable_cmc else None
    cut_detector = CameraCutDetector() if enable_cut_detection else None
    reid = PlayerReID() if enable_reid else None

    # 2.5 Match Event Detector (Milestone 11)
    event_detector = EventDetector(fps=props["fps"]) if enable_events else None

    # 3. Pitch Detector
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

    # 4. Tracker
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

    # 5. Ball Tracker & Possession Engine
    ball_tracker = None
    if enable_ball_tracking:
        ball_tracker = BallTracker(
            fps=props["fps"],
            max_interpolation_gap=6,
            max_speed_kmh=140.0,
            possession_radius_m=1.8,
            trail_length=25,
            hysteresis_frames=2,
        )

    # 6. Team Classifier
    team_classifier = None
    if enable_team:
        team_cfg = config.get("team", {})
        team_classifier = TeamClassifier(
            n_teams=team_cfg.get("n_teams", 2),
            history_window=team_cfg.get("history_window", 20),
            referee_dist_threshold=team_cfg.get("referee_dist_threshold", 85.0),
        )

    # 7. Homography & 2D Radar
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

    # 8. Speed Estimator
    speed_estimator = None
    if enable_speed:
        analytics_cfg = config.get("analytics", {})
        speed_estimator = SpeedEstimator(
            fps=props["fps"],
            window_size=analytics_cfg.get("window_size", 18),
            max_speed_kmh=analytics_cfg.get("max_realistic_speed_kmh", 38.0),
            min_speed_kmh=analytics_cfg.get("min_speed_threshold_kmh", 2.5),
        )

    # 9. Tactical Spatial Control & Heatmaps
    spatial_control = None
    heatmap_gen = None
    if enable_tactics:
        spatial_control = SpatialControl()
        heatmap_gen = HeatmapGenerator()

    # 10. Annotator
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
        draw_ball_trail=enable_ball_tracking,
        draw_possession=enable_ball_tracking,
        draw_camera_motion=enable_cmc,
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
    pan_events = {"PAN RIGHT": 0, "PAN LEFT": 0, "STATIC": 0}
    cut_events = []
    last_possession_result: Optional[PossessionResult] = None

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

            # Step 4: Camera Motion Estimation (GME)
            camera_motion = None
            camera_transform = None
            if camera_compensator is not None:
                camera_motion = camera_compensator.estimate_motion(
                    proc_frame,
                    detections=detections,
                    frame_idx=frame_idx,
                )
                camera_transform = camera_motion.transform_matrix
                pan_events[camera_motion.pan_direction] = pan_events.get(camera_motion.pan_direction, 0) + 1

            # Step 4.5: Camera Cut Detection (Milestone 10)
            cut_result = None
            if cut_detector is not None:
                cut_result = cut_detector.detect_cut(
                    proc_frame,
                    frame_idx=frame_idx,
                    flow_inlier_ratio=camera_motion.confidence if camera_motion else None,
                )
                if cut_result.is_cut:
                    cut_events.append(frame_idx)

            # Step 5: Multi-Object Tracking (ByteTrack + CMC + Cut-Aware Re-ID)
            if tracker is not None:
                processed_results = tracker.update(
                    detections,
                    camera_transform=camera_transform,
                    is_cut=cut_result.is_cut if cut_result else False,
                    reid=reid,
                    frame=frame,
                )
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

            # Step 6: Homography & Metric Coordinates Calculation
            player_positions_m = np.empty((0, 2), dtype=np.float32)
            ball_pos_m = None
            H_matrix = None

            if pitch_result.pitch_area_ratio >= 0.20 and num_players > 0:
                homography_res = calibrator.estimate_broadcast_homography(
                    frame_shape=(props["height"], props["width"]),
                    top_touchline_y=245.0,
                    bottom_touchline_y=745.0,
                    halfway_x=960.0,
                    center_y=500.0,
                )

                if homography_res.is_valid:
                    H_matrix = homography_res.H
                    feet_pts = players_res.get_foot_positions()
                    player_positions_m = calibrator.image_to_pitch(feet_pts, homography_res.H)

                    if has_ball:
                        ball_box = ball_res.xyxy[0]
                        ball_center = np.array([[(ball_box[0] + ball_box[2]) / 2.0, (ball_box[1] + ball_box[3]) / 2.0]], dtype=np.float32)
                        ball_m = calibrator.image_to_pitch(ball_center, homography_res.H)
                        if len(ball_m) > 0:
                            ball_pos_m = (float(ball_m[0, 0]), float(ball_m[0, 1]))

            # Step 7: Team, Goalkeeper, Referee, and Coach Classification
            team_result = None
            player_team_ids = None
            if team_classifier is not None:
                team_result = team_classifier.classify_frame(
                    frame=frame,
                    detections=processed_results,
                    positions_m=player_positions_m if len(player_positions_m) == len(processed_results.xyxy) else None,
                )
                if team_result is not None:
                    player_indices = np.where(processed_results.class_ids == 0)[0]
                    player_team_ids = team_result.team_ids[player_indices] if len(player_indices) > 0 else np.empty((0,), dtype=int)

            # Step 8: Kalman Ball Tracking, Gap Interpolation & Player Possession Assignment
            ball_state = None
            possession_result = None
            if ball_tracker is not None:
                ball_state, possession_result = ball_tracker.update(
                    detections=processed_results,
                    homography_matrix=H_matrix,
                    player_positions_m=player_positions_m if len(player_positions_m) > 0 else None,
                    player_track_ids=players_res.tracker_ids if len(players_res.tracker_ids) > 0 else None,
                    player_team_ids=player_team_ids,
                    frame_idx=frame_idx,
                )
                if ball_state is not None and ball_state.position_m is not None:
                    ball_pos_m = ball_state.position_m
                last_possession_result = possession_result

            # Step 8.5: Discrete Match Event Recognition (Milestone 11)
            if event_detector is not None:
                event_detector.update(
                    ball_state=ball_state,
                    possession_result=possession_result,
                    player_positions_m=player_positions_m if len(player_positions_m) > 0 else None,
                    player_track_ids=players_res.tracker_ids if len(players_res.tracker_ids) > 0 else None,
                    player_team_ids=player_team_ids,
                    frame_idx=frame_idx,
                )

            # Step 9: Speed & Distance Estimation
            player_metrics = {}
            team_summary = {}
            if speed_estimator is not None and len(player_positions_m) > 0:
                player_metrics = speed_estimator.update(
                    track_ids=players_res.tracker_ids,
                    positions_m=player_positions_m,
                    frame_idx=frame_idx,
                )
                team_summary = speed_estimator.get_team_summary()

            # Step 10: Tactical Spatial Control & Heatmaps
            tactical_spatial_result = None
            if spatial_control is not None and team_result is not None and len(player_positions_m) > 0:
                if player_team_ids is not None and len(player_team_ids) == len(player_positions_m):
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

            # Step 11: Visual Annotation & Telemetry HUD
            annotated_frame = annotator.annotate(
                frame=frame,
                detections=processed_results,
                pitch_result=pitch_result,
                team_result=team_result,
                tactical_spatial_result=tactical_spatial_result,
                player_metrics=player_metrics,
                team_summary=team_summary,
                ball_state=ball_state,
                possession_result=possession_result,
                camera_motion=camera_motion,
                cut_result=cut_result,
                reid_count=reid.total_reassignments if reid is not None else 0,
                active_event=event_detector.active_event if event_detector is not None else None,
                ball_trail=ball_tracker.trail if ball_tracker is not None else None,
                fps=fps_inst,
                frame_idx=frame_idx + 1,
                total_frames=total_frames,
                device_name=device_label,
            )

            # Step 12: 2D Tactical Radar Minimap Overlay
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
                    possession_result=possession_result,
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

    # Step 13: Export Tactical Positional Heatmaps
    if heatmap_gen is not None:
        heatmap_dir = config.get("tactics", {}).get("heatmap_output_dir", "outputs/heatmaps")
        heatmap_gen.export_all_heatmaps(heatmap_dir)

    total_time = time.time() - start_time
    avg_fps = len(frame_times) / total_time if total_time > 0 else 0.0
    avg_players = float(np.mean(player_counts)) if player_counts else 0.0
    final_team_summary = speed_estimator.get_team_summary() if speed_estimator else {}
    mean_ctrl_a = float(np.mean(avg_team_a_control)) if avg_team_a_control else 50.0
    mean_ctrl_b = float(np.mean(avg_team_b_control)) if avg_team_b_control else 50.0

    # Compile final possession summary
    poss_summary = {}
    if last_possession_result is not None:
        top_carrier = None
        if last_possession_result.player_possession_counts:
            top_carrier = max(last_possession_result.player_possession_counts.items(), key=lambda x: x[1])[0]

        poss_summary = {
            "team_a_possession_pct": last_possession_result.team_a_possession_pct,
            "team_b_possession_pct": last_possession_result.team_b_possession_pct,
            "contested_pct": last_possession_result.contested_pct,
            "turnover_count": last_possession_result.turnover_count,
            "top_carrier_id": top_carrier,
            "player_possession_counts": last_possession_result.player_possession_counts,
        }

    # Compile event summary
    event_summary_dict = {}
    if event_detector is not None:
        ev_sum = event_detector.get_summary()
        event_summary_dict = {
            "total_events": ev_sum.total_events,
            "team_a_passes": f"{ev_sum.completed_passes_a}/{ev_sum.total_passes_a} ({ev_sum.pass_accuracy_a_pct}%)",
            "team_b_passes": f"{ev_sum.completed_passes_b}/{ev_sum.total_passes_b} ({ev_sum.pass_accuracy_b_pct}%)",
            "team_a_shots": ev_sum.total_shots_a,
            "team_b_shots": ev_sum.total_shots_b,
            "team_a_interceptions": ev_sum.total_interceptions_a,
            "team_b_interceptions": ev_sum.total_interceptions_b,
            "team_a_tackles": ev_sum.total_tackles_a,
            "team_b_tackles": ev_sum.total_tackles_b,
            "timeline": ev_sum.events_timeline,
        }

    print(f"\n=======================================================")
    print(f" Processing Complete (Milestone 11)!")
    print(f"=======================================================")
    print(f" Total Elapsed Time     : {total_time:.2f} seconds")
    print(f" Average Speed          : {avg_fps:.2f} FPS")
    print(f" Avg On-Pitch Players   : {avg_players:.1f}")
    print(f" Pitch Locked           : {pitch_detected_count}/{total_frames} frames ({pitch_detected_count/max(1,total_frames)*100:.1f}%)")
    print(f" Camera Motion Events   : {pan_events.get('PAN RIGHT', 0)} Pan Right, {pan_events.get('PAN LEFT', 0)} Pan Left, {pan_events.get('STATIC', 0)} Static")
    print(f" Camera Cut Events      : {len(cut_events)} cuts detected (Frames: {cut_events[:8]}{'...' if len(cut_events) > 8 else ''})")
    if reid is not None:
        print(f" Re-ID Reassignments   : {reid.total_reassignments} player identities preserved across cuts")
    if event_detector is not None:
        ev_sum = event_detector.get_summary()
        print(f" Match Events Detected  : {ev_sum.total_events} events logged")
        print(f"   - Team A Passes      : {ev_sum.completed_passes_a}/{ev_sum.total_passes_a} ({ev_sum.pass_accuracy_a_pct}%)")
        print(f"   - Team B Passes      : {ev_sum.completed_passes_b}/{ev_sum.total_passes_b} ({ev_sum.pass_accuracy_b_pct}%)")
        print(f"   - Shots on Goal      : Team A: {ev_sum.total_shots_a} | Team B: {ev_sum.total_shots_b}")
        print(f"   - Interceptions      : Team A: {ev_sum.total_interceptions_a} | Team B: {ev_sum.total_interceptions_b}")
        print(f"   - Tackles / Duels    : Team A: {ev_sum.total_tackles_a} | Team B: {ev_sum.total_tackles_b}")
    print(f" Team A Space Dominance : {mean_ctrl_a:.1f}% pitch space control")
    print(f" Team B Space Dominance : {mean_ctrl_b:.1f}% pitch space control")
    if poss_summary:
        print(f" Team A Ball Possession : {poss_summary.get('team_a_possession_pct', 50.0):.1f}%")
        print(f" Team B Ball Possession : {poss_summary.get('team_b_possession_pct', 50.0):.1f}%")
        print(f" Possession Turnovers   : {poss_summary.get('turnover_count', 0)} turnovers")
        print(f" Top Ball Carrier Track : #{poss_summary.get('top_carrier_id', 'N/A')}")
    if enable_tracking:
        print(f" Total Unique Tracks    : {len(unique_track_ids)} unique IDs assigned")
    print(f" Peak Sprint Speed      : {final_team_summary.get('max_speed_kmh', 0.0)} km/h")
    print(f" Total Distance Run     : {final_team_summary.get('total_distance_km', 0.0)} km")
    print(f" Ball Detected In       : {ball_detected_count}/{total_frames} frames ({ball_detected_count/max(1,total_frames)*100:.1f}%)")
    print(f" Output Video           : {output_path}")
    print(f" Heatmaps Saved To      : outputs/heatmaps/")
    print(f"=======================================================")

    log_dir = Path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "source": str(source),
        "output": output_path,
        "model": model_name,
        "imgsz": imgsz,
        "device": device_label,
        "camera_motion_compensation_enabled": enable_cmc,
        "camera_pan_events": pan_events,
        "camera_cut_detection_enabled": enable_cut_detection,
        "total_camera_cuts": len(cut_events),
        "camera_cut_frames": cut_events,
        "player_reid_enabled": enable_reid,
        "reid_total_reassignments": reid.total_reassignments if reid is not None else 0,
        "match_events_enabled": enable_events,
        "match_events_summary": event_summary_dict,
        "team_classification_enabled": enable_team,
        "tactical_space_control_enabled": enable_tactics,
        "ball_possession_summary": poss_summary,
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

    # Save dedicated match events timeline log
    if event_detector is not None:
        events_path = log_dir / f"{source.stem}_match_events.json"
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump(event_summary_dict, f, indent=2)

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
        enable_ball_tracking=not args.no_ball_track,
        enable_cmc=not args.no_cmc,
        enable_cut_detection=not args.no_cut_detect,
        enable_reid=not args.no_reid,
        enable_events=not args.no_events,
        draw_pitch_lines=args.draw_pitch_lines,
        draw_pitch_boundary=args.draw_pitch_boundary,
        draw_trails=not args.no_trails,
        draw_hud=not args.no_hud,
    )
