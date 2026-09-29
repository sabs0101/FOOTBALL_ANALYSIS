"""
Asynchronous Multi-Threaded Tactical Football Analysis Pipeline.
Decouples Frame Ingestion, GPU Inference & Tracking, Tactical Analytics,
and Visual Egress/Streaming into 4 concurrent ring-buffered worker threads.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.analytics.events import EventDetector, MatchEvent
from src.analytics.speed_distance import SpeedEstimator, PlayerMetrics
from src.analytics.xg_xt import TacticalAdvancedEngine, AdvancedTacticsSummary
from src.calibration.camera_motion import CameraMotionCompensator, CameraMotionResult
from src.calibration.homography import PitchHomography, HomographyResult
from src.calibration.template import PitchTemplate
from src.detection.detector import PlayerDetector, DetectionResult
from src.pitch.detector import PitchDetector, PitchResult
from src.preprocessing.preprocessor import FramePreprocessor
from src.tactics.heatmaps import HeatmapGenerator
from src.tactics.spatial import SpatialControl, TacticalSpatialResult
from src.team.classifier import TeamClassifier, TeamResult
from src.tracking.ball_tracker import BallTracker, BallState, PossessionResult
from src.tracking.cut_detector import CameraCutDetector, CutDetectionResult
from src.tracking.reid import PlayerReID
from src.tracking.tracker import PlayerTracker
from src.utils.config import get_device, load_config
from src.utils.video import VideoReader, VideoWriter, get_video_properties
from src.visualization.annotator import VideoAnnotator
from src.visualization.radar import TacticalRadar
from src.visualization.passing_network import PassingNetworkVisualizer



@dataclass
class FramePacket:
    """Stage 1: Raw ingested video frame."""
    frame_idx: int
    raw_frame: np.ndarray
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrackingPacket:
    """Stage 2: Detection, CMC, Cut, and Tracking Results."""
    frame_idx: int
    raw_frame: np.ndarray
    proc_frame: np.ndarray
    detections: DetectionResult
    camera_motion: Optional[CameraMotionResult]
    camera_transform: Optional[np.ndarray]
    cut_result: Optional[CutDetectionResult]
    pitch_result: PitchResult
    is_keyframe: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class AnalyticsPacket:
    """Stage 3: Full Pitch Homography, Team ID, Kinematics, Tactics, and Events."""
    frame_idx: int
    raw_frame: np.ndarray
    detections: DetectionResult
    pitch_result: PitchResult
    team_result: Optional[TeamResult]
    tactical_spatial_result: Optional[TacticalSpatialResult]
    player_metrics: Dict[int, PlayerMetrics]
    team_summary: Dict[str, Any]
    ball_state: Optional[BallState]
    possession_result: Optional[PossessionResult]
    camera_motion: Optional[CameraMotionResult]
    cut_result: Optional[CutDetectionResult]
    active_event: Optional[MatchEvent]
    player_positions_m: np.ndarray
    ball_pos_m: Optional[Tuple[float, float]]
    xg_scores: Optional[Tuple[float, float]] = None
    timestamp: float = field(default_factory=time.time)



@dataclass
class PipelineMetrics:
    """Aggregated performance and tactical summary metrics."""
    total_frames: int = 0
    processed_frames: int = 0
    elapsed_time_s: float = 0.0
    average_fps: float = 0.0
    stage1_ingest_ms: float = 0.0
    stage2_inference_ms: float = 0.0
    stage3_analytics_ms: float = 0.0
    stage4_egress_ms: float = 0.0
    unique_track_ids_count: int = 0
    camera_cuts_count: int = 0
    events_count: int = 0
    possession_team_a_pct: float = 0.0
    possession_team_b_pct: float = 0.0
    turnovers_count: int = 0
    top_speed_kmh: float = 0.0
    total_distance_team_a_km: float = 0.0
    total_distance_team_b_km: float = 0.0


class AsyncTacticalPipeline:
    """
    High-Throughput Asynchronous Multi-Threaded Tactical Analysis Pipeline.
    
    Architecture (4 Concurrent Ring-Buffered Worker Threads):
      [Video Ingestion] -> _ingest_queue -> [GPU Inference & Tracking]
                        -> _inference_queue -> [Tactical Analytics & Events]
                        -> _analytics_queue -> [Visual Egress & Live Streaming]
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        on_frame_processed: Optional[Callable[[int, Dict[str, Any], np.ndarray], None]] = None,
        on_progress: Optional[Callable[[int, int, float], None]] = None,
        queue_size: int = 32,
    ):
        """
        Initialize Async Tactical Pipeline with configuration options.
        """
        self.config = config or load_config("config.yaml")
        self.options = options or {}
        self.on_frame_processed = on_frame_processed
        self.on_progress = on_progress
        self.queue_size = queue_size

        # Operational Flags
        self.enable_clahe = self.options.get("clahe", True)
        self.enable_cmc = self.options.get("cmc", True)
        self.enable_cut_detect = self.options.get("cut_detect", True)
        self.enable_reid = self.options.get("reid", True)
        self.enable_pitch_filter = self.options.get("pitch_filter", True)
        self.enable_team = self.options.get("team", True)
        self.enable_radar = self.options.get("radar", True)
        self.enable_speed = self.options.get("speed", True)
        self.enable_tactics = self.options.get("tactics", True)
        self.enable_heatmaps = self.options.get("heatmaps", True)
        self.enable_ball_tracking = self.options.get("ball_track", True)
        self.enable_events = self.options.get("events", True)
        self.enable_advanced_tactics = self.options.get("advanced_tactics", True)

        # Performance Decoupling Parameters (Milestone 14)
        self.detector_interval = int(self.options.get("detector_interval", 1))
        self.sparse_gme = bool(self.options.get("sparse_gme", True))
        self.homography_interval = int(self.options.get("homography_interval", 25))

        # Threading infrastructure
        self._stop_event = threading.Event()
        self._ingest_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._inference_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._analytics_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)

        self._threads: List[threading.Thread] = []
        self._exceptions: List[Exception] = []
        self._results: Dict[str, Any] = {}

        # Worker latencies (EMA smoothed ms)
        self.stage_latencies = {
            "ingest_ms": 0.0,
            "inference_ms": 0.0,
            "analytics_ms": 0.0,
            "egress_ms": 0.0,
        }

        # Module references (instantiated in _init_modules)
        self.preprocessor = None
        self.detector = None
        self.camera_compensator = None
        self.cut_detector = None
        self.reid = None
        self.pitch_detector = None
        self.tracker = None
        self.calibrator = None
        self.team_classifier = None
        self.ball_tracker = None
        self.event_detector = None
        self.speed_estimator = None
        self.spatial_control = None
        self.heatmap_gen = None
        self.advanced_engine = None
        self.tactical_radar = None
        self.passing_visualizer = None
        self.annotator = None

    def _init_modules(self, props: Dict[str, Any], preferred_device: str):

        """Instantiate domain modules for worker stages."""
        fps = props["fps"]
        w = props["width"]
        h = props["height"]

        # Stage 2 Modules
        self.preprocessor = FramePreprocessor(enable_clahe=self.enable_clahe) if self.enable_clahe else None
        
        model_name = self.options.get("model", self.config.get("detection", {}).get("model", "models/yolov8m.pt"))
        conf_thresh = float(self.options.get("conf", self.config.get("detection", {}).get("confidence_threshold", 0.18)))
        imgsz = int(self.options.get("imgsz", self.config.get("detection", {}).get("image_size", 1280)))
        
        self.detector = PlayerDetector(
            model_name=model_name,
            device=preferred_device,
            conf_threshold=conf_thresh,
            imgsz=imgsz,
        )
        self.camera_compensator = CameraMotionCompensator(sparse_mode=self.sparse_gme) if self.enable_cmc else None
        self.cut_detector = CameraCutDetector() if self.enable_cut_detect else None
        self.reid = PlayerReID() if self.enable_reid else None
        self.pitch_detector = PitchDetector()
        self.tracker = PlayerTracker(frame_rate=int(fps))

        # Stage 3 Modules
        self.calibrator = PitchHomography()
        self.team_classifier = TeamClassifier() if self.enable_team else None
        self.ball_tracker = BallTracker(fps=fps) if self.enable_ball_tracking else None
        self.event_detector = EventDetector(fps=fps) if self.enable_events else None
        self.speed_estimator = SpeedEstimator(fps=fps) if self.enable_speed else None
        self.spatial_control = SpatialControl() if self.enable_tactics else None
        self.heatmap_gen = HeatmapGenerator() if self.enable_heatmaps else None
        self.advanced_engine = TacticalAdvancedEngine() if self.enable_advanced_tactics else None

        # Stage 4 Modules
        radar_cfg = self.config.get("radar", {})
        self.tactical_radar = TacticalRadar(
            radar_width=radar_cfg.get("radar_width", 380),
            radar_height=radar_cfg.get("radar_height", 245),
        ) if self.enable_radar else None
        self.passing_visualizer = PassingNetworkVisualizer() if self.enable_advanced_tactics else None


        self.annotator = VideoAnnotator(
            box_thickness=self.config["visualization"].get("box_thickness", 2),
            font_scale=self.config["visualization"].get("text_scale", 0.45),
            draw_conf=self.config["visualization"].get("draw_confidences", False),
            draw_hud=self.config["visualization"].get("draw_hud", True),
            draw_tracks=self.config["visualization"].get("draw_tracks", True),
            draw_trails=self.config["visualization"].get("draw_trails", False),
            draw_speed=self.enable_speed,
            draw_team=self.enable_team,
            draw_pitch_boundary=False,
            draw_pitch_lines=False,
            draw_ball_trail=self.config["visualization"].get("draw_ball_trail", False),
            draw_possession=self.enable_ball_tracking,
            draw_camera_motion=self.enable_cmc,
        )

    # -------------------------------------------------------------------------
    # Worker Stage 1: Hardware Video Ingestion
    # -------------------------------------------------------------------------
    def _ingest_worker(self, source_path: str):
        """Worker 1: Ingests frames and feeds the thread-safe ring buffer."""
        try:
            reader = VideoReader(str(source_path))
            for frame_idx, raw_frame in reader:
                if self._stop_event.is_set():
                    break
                t0 = time.perf_counter()
                packet = FramePacket(frame_idx=frame_idx, raw_frame=raw_frame)
                
                # Push with timeout to check for shutdown
                while not self._stop_event.is_set():
                    try:
                        self._ingest_queue.put(packet, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                
                t_ingest = (time.perf_counter() - t0) * 1000.0
                self.stage_latencies["ingest_ms"] = 0.9 * self.stage_latencies["ingest_ms"] + 0.1 * t_ingest

            reader.release()
        except Exception as e:
            self._exceptions.append(e)
            self._stop_event.set()
        finally:
            # Send sentinel poison pill to next stage without blocking if stopped
            try:
                while not self._stop_event.is_set():
                    try:
                        self._ingest_queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Worker Stage 2: GPU Inference & Tracking
    # -------------------------------------------------------------------------
    def _inference_worker(self):
        """Worker 2: Runs Camera Motion, Cuts, Pitch Lines, and YOLO Detection / ByteTrack."""
        last_pitch_result: Optional[PitchResult] = None
        try:
            while not self._stop_event.is_set():
                try:
                    packet: Optional[FramePacket] = self._ingest_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if packet is None:
                    # End of stream sentinel
                    break

                t0 = time.perf_counter()
                frame_idx = packet.frame_idx
                raw_frame = packet.raw_frame

                # Step 1: Preprocessing
                proc_frame = self.preprocessor.process(raw_frame).frame if self.preprocessor else raw_frame

                # Step 2: Camera Motion Estimation (GME)
                camera_motion = None
                camera_transform = None
                if self.camera_compensator is not None:
                    camera_motion = self.camera_compensator.estimate_motion(
                        proc_frame,
                        detections=None,
                        frame_idx=frame_idx,
                    )
                    camera_transform = camera_motion.transform_matrix

                # Step 3: Camera Cut Detection
                cut_result = None
                is_cut = False
                if self.cut_detector is not None:
                    cut_result = self.cut_detector.detect_cut(
                        proc_frame,
                        frame_idx=frame_idx,
                        flow_inlier_ratio=camera_motion.confidence if camera_motion else None,
                    )
                    is_cut = cut_result.is_cut

                # Step 4: Pitch & Line Detection (Interval-based)
                need_full_pitch = (
                    last_pitch_result is None
                    or is_cut
                    or (self.homography_interval <= 1)
                    or (frame_idx % self.homography_interval == 0)
                )
                if need_full_pitch:
                    pitch_result = self.pitch_detector.detect_lines(proc_frame)
                    last_pitch_result = pitch_result
                else:
                    pitch_result = last_pitch_result

                # Step 5: Temporal Decoupling: Keyframe YOLO vs Intervening ByteTrack
                is_keyframe = (self.detector_interval <= 1) or (frame_idx % self.detector_interval == 0) or is_cut

                if self.tracker is not None:
                    if is_keyframe:
                        detections = self.detector.detect(proc_frame, frame_idx=frame_idx)
                        if self.enable_pitch_filter and pitch_result.pitch_area_ratio >= 0.20:
                            detections = self.pitch_detector.filter_detections_on_pitch(detections, pitch_result.mask)
                        processed_results = self.tracker.update(
                            detections,
                            camera_transform=camera_transform,
                            is_cut=is_cut,
                            reid=self.reid,
                            frame=raw_frame,
                        )
                    else:
                        processed_results = self.tracker.propagate_intervening(
                            camera_transform=camera_transform,
                            frame_idx=frame_idx,
                        )
                else:
                    if is_keyframe:
                        detections = self.detector.detect(proc_frame, frame_idx=frame_idx)
                        if self.enable_pitch_filter and pitch_result.pitch_area_ratio >= 0.20:
                            detections = self.pitch_detector.filter_detections_on_pitch(detections, pitch_result.mask)
                        processed_results = detections
                    else:
                        processed_results = DetectionResult(
                            xyxy=np.empty((0, 4), dtype=np.float32),
                            confidences=np.empty((0,), dtype=np.float32),
                            class_ids=np.empty((0,), dtype=int),
                            class_names=[],
                            frame_idx=frame_idx,
                        )

                tracking_packet = TrackingPacket(
                    frame_idx=frame_idx,
                    raw_frame=raw_frame,
                    proc_frame=proc_frame,
                    detections=processed_results,
                    camera_motion=camera_motion,
                    camera_transform=camera_transform,
                    cut_result=cut_result,
                    pitch_result=pitch_result,
                    is_keyframe=is_keyframe,
                )

                while not self._stop_event.is_set():
                    try:
                        self._inference_queue.put(tracking_packet, timeout=0.1)
                        break
                    except queue.Full:
                        continue

                t_infer = (time.perf_counter() - t0) * 1000.0
                self.stage_latencies["inference_ms"] = 0.9 * self.stage_latencies["inference_ms"] + 0.1 * t_infer

        except Exception as e:
            self._exceptions.append(e)
            self._stop_event.set()
        finally:
            try:
                while not self._stop_event.is_set():
                    try:
                        self._inference_queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Worker Stage 3: Tactical Analytics Engine
    # -------------------------------------------------------------------------
    def _analytics_worker(self, height: int, width: int):
        """Worker 3: Homography, Metric Projection, Team ID, Kinematics, Space, Ball & Events."""
        last_homography_res: Optional[HomographyResult] = None
        H_matrix = None

        try:
            # Initial homography estimation
            initial_homography = self.calibrator.estimate_broadcast_homography((height, width))
            last_homography_res = initial_homography
            if initial_homography.is_valid:
                H_matrix = initial_homography.H

            while not self._stop_event.is_set():
                try:
                    packet: Optional[TrackingPacket] = self._inference_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if packet is None:
                    break

                t0 = time.perf_counter()
                frame_idx = packet.frame_idx
                raw_frame = packet.raw_frame
                detections = packet.detections
                pitch_result = packet.pitch_result
                camera_transform = packet.camera_transform
                cut_result = packet.cut_result
                is_cut = cut_result.is_cut if cut_result else False

                players_res = detections.get_players()
                num_players = len(players_res.xyxy)
                ball_res = detections.get_ball()
                has_ball = len(ball_res.xyxy) > 0

                # 1. Homography & Metric Space Projection
                player_positions_m = np.empty((0, 2), dtype=np.float32)
                ball_pos_m = None

                need_full_pitch = (
                    last_homography_res is None
                    or is_cut
                    or (self.homography_interval <= 1)
                    or (frame_idx % self.homography_interval == 0)
                )

                if pitch_result.pitch_area_ratio >= 0.20:
                    if need_full_pitch or last_homography_res is None:
                        homography_res = self.calibrator.estimate_broadcast_homography(
                            frame_shape=(height, width),
                            top_touchline_y=245.0,
                            bottom_touchline_y=745.0,
                            halfway_x=960.0,
                            center_y=500.0,
                        )
                        last_homography_res = homography_res
                    elif camera_transform is not None and last_homography_res is not None:
                        homography_res = self.calibrator.propagate_homography(last_homography_res, camera_transform)
                        last_homography_res = homography_res
                    else:
                        homography_res = last_homography_res

                    if homography_res is not None and homography_res.is_valid:
                        H_matrix = homography_res.H
                        if num_players > 0:
                            feet_pts = players_res.get_foot_positions()
                            player_positions_m = self.calibrator.image_to_pitch(feet_pts, homography_res.H)

                        if has_ball:
                            ball_box = ball_res.xyxy[0]
                            ball_center = np.array([[(ball_box[0] + ball_box[2]) / 2.0, (ball_box[1] + ball_box[3]) / 2.0]], dtype=np.float32)
                            ball_m = self.calibrator.image_to_pitch(ball_center, homography_res.H)
                            if len(ball_m) > 0:
                                ball_pos_m = (float(ball_m[0, 0]), float(ball_m[0, 1]))

                # 2. Team & Role Classification
                team_result = None
                player_team_ids = None
                if self.team_classifier is not None:
                    team_result = self.team_classifier.classify_frame(
                        frame=raw_frame,
                        detections=detections,
                        positions_m=player_positions_m if len(player_positions_m) > 0 else None,
                    )
                    if team_result is not None:
                        player_indices = np.where(detections.class_ids == 0)[0]
                        player_team_ids = team_result.team_ids[player_indices] if len(player_indices) > 0 else np.empty((0,), dtype=int)

                # 3. Kalman Ball Tracking & Possession Assignment
                ball_state = None
                possession_result = None
                player_tids = players_res.tracker_ids if (players_res.tracker_ids is not None and len(players_res.tracker_ids) > 0) else None

                if self.ball_tracker is not None:
                    ball_state, possession_result = self.ball_tracker.update(
                        detections=detections,
                        homography_matrix=H_matrix,
                        player_positions_m=player_positions_m if len(player_positions_m) > 0 else None,
                        player_track_ids=player_tids,
                        player_team_ids=player_team_ids,
                        frame_idx=frame_idx,
                    )
                    if ball_state is not None and ball_state.position_m is not None:
                        ball_pos_m = ball_state.position_m

                # 4. Discrete Match Events Recognition (Milestone 11)
                active_event = None
                if self.event_detector is not None:
                    self.event_detector.update(
                        ball_state=ball_state,
                        possession_result=possession_result,
                        player_positions_m=player_positions_m if len(player_positions_m) > 0 else None,
                        player_track_ids=player_tids,
                        player_team_ids=player_team_ids,
                        frame_idx=frame_idx,
                    )
                    active_event = self.event_detector.active_event

                # 5. Speed Kinematics & Distance Estimation
                player_metrics: Dict[int, PlayerMetrics] = {}
                team_summary: Dict[str, Any] = {}
                if self.speed_estimator is not None and player_tids is not None and len(player_positions_m) > 0 and len(player_tids) == len(player_positions_m):
                    player_metrics = self.speed_estimator.update(
                        track_ids=player_tids,
                        positions_m=player_positions_m,
                        frame_idx=frame_idx,
                    )
                    team_summary = self.speed_estimator.get_team_summary()

                # 6. Tactical Spatial Control & Heatmaps
                tactical_spatial_result = None
                if self.spatial_control is not None and team_result is not None and len(player_positions_m) > 0:
                    if player_team_ids is not None and len(player_team_ids) == len(player_positions_m):
                        tactical_spatial_result = self.spatial_control.analyze_frame(
                            positions_m=player_positions_m,
                            team_ids=player_team_ids,
                        )
                        if self.heatmap_gen is not None and player_tids is not None and len(player_tids) == len(player_positions_m):
                            self.heatmap_gen.add_positions(
                                track_ids=player_tids,
                                positions_m=player_positions_m,
                                team_ids=player_team_ids,
                            )

                # 7. Advanced Tactical Intelligence: xG, xT, and Passing Networks (Milestone 17)
                xg_scores = None
                if self.advanced_engine is not None:
                    if len(player_positions_m) > 0 and player_tids is not None and player_team_ids is not None:
                        self.advanced_engine.update_player_positions(
                            player_track_ids=player_tids,
                            player_team_ids=player_team_ids,
                            player_positions_m=player_positions_m,
                        )
                    if active_event is not None and getattr(active_event, "frame_idx", -1) == frame_idx:
                        etype = getattr(active_event, "event_type", "")
                        if etype == "SHOT":
                            self.advanced_engine.record_shot_event(
                                shooter_id=active_event.primary_player_id,
                                team_id=active_event.team_id,
                                shot_pos_m=active_event.start_pos_m,
                                frame_idx=frame_idx,
                                timestamp_s=active_event.timestamp_s,
                                shot_speed_kmh=active_event.speed_kmh,
                                defenders_m=player_positions_m if len(player_positions_m) > 0 else None,
                            )
                        elif etype == "PASS" and active_event.is_successful:
                            self.advanced_engine.record_pass_action(
                                passer_id=active_event.primary_player_id,
                                receiver_id=active_event.secondary_player_id,
                                team_id=active_event.team_id,
                                start_pos_m=active_event.start_pos_m,
                                end_pos_m=active_event.end_pos_m,
                                frame_idx=frame_idx,
                                timestamp_s=active_event.timestamp_s,
                                speed_kmh=active_event.speed_kmh,
                            )
                    xg_scores = (
                        round(self.advanced_engine.total_xg_team_a, 2),
                        round(self.advanced_engine.total_xg_team_b, 2),
                    )

                analytics_packet = AnalyticsPacket(
                    frame_idx=frame_idx,
                    raw_frame=raw_frame,
                    detections=detections,
                    pitch_result=pitch_result,
                    team_result=team_result,
                    tactical_spatial_result=tactical_spatial_result,
                    player_metrics=player_metrics,
                    team_summary=team_summary,
                    ball_state=ball_state,
                    possession_result=possession_result,
                    camera_motion=packet.camera_motion,
                    cut_result=cut_result,
                    active_event=active_event,
                    player_positions_m=player_positions_m,
                    ball_pos_m=ball_pos_m,
                    xg_scores=xg_scores,
                )


                while not self._stop_event.is_set():
                    try:
                        self._analytics_queue.put(analytics_packet, timeout=0.1)
                        break
                    except queue.Full:
                        continue

                t_analytics = (time.perf_counter() - t0) * 1000.0
                self.stage_latencies["analytics_ms"] = 0.9 * self.stage_latencies["analytics_ms"] + 0.1 * t_analytics

        except Exception as e:
            self._exceptions.append(e)
            self._stop_event.set()
        finally:
            try:
                while not self._stop_event.is_set():
                    try:
                        self._analytics_queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Worker Stage 4: Visual Egress, Streaming & Metrics Hub
    # -------------------------------------------------------------------------
    def _egress_worker(self, output_path: Optional[str], fps: float, width: int, height: int, total_frames: int):
        """Worker 4: Renders Tactical Annotations/Radar, Writes Video, and Dispatches Telemetry."""
        writer: Optional[VideoWriter] = None
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = VideoWriter(output_path=output_path, fps=fps, width=width, height=height, codec="h264")

        start_time = time.time()
        processed_count = 0
        cur_fps = 0.0
        unique_track_ids = set()
        pan_counts = {"PAN RIGHT": 0, "PAN LEFT": 0, "STATIC": 0}
        cut_frames = []
        team_a_control_list = []
        team_b_control_list = []

        try:
            while not self._stop_event.is_set():
                try:
                    packet: Optional[AnalyticsPacket] = self._analytics_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if packet is None:
                    break

                t0 = time.perf_counter()
                frame_idx = packet.frame_idx
                raw_frame = packet.raw_frame
                detections = packet.detections
                players_res = detections.get_players()
                num_players = len(players_res.xyxy)

                if players_res.tracker_ids is not None:
                    for tid in players_res.tracker_ids:
                        if tid >= 0:
                            unique_track_ids.add(int(tid))

                if packet.camera_motion:
                    pan_dir = packet.camera_motion.pan_direction
                    pan_counts[pan_dir] = pan_counts.get(pan_dir, 0) + 1

                if packet.cut_result and packet.cut_result.is_cut:
                    cut_frames.append(frame_idx)

                if packet.tactical_spatial_result:
                    team_a_control_list.append(packet.tactical_spatial_result.team_a_control_pct)
                    team_b_control_list.append(packet.tactical_spatial_result.team_b_control_pct)

                processed_count += 1
                elapsed_total = max(0.001, time.time() - start_time)
                cur_fps = round(processed_count / elapsed_total, 1)

                # Visual HUD Annotation
                annotated_frame = self.annotator.annotate(
                    frame=raw_frame,
                    detections=detections,
                    pitch_result=packet.pitch_result,
                    team_result=packet.team_result,
                    tactical_spatial_result=packet.tactical_spatial_result,
                    player_metrics=packet.player_metrics,
                    team_summary=packet.team_summary,
                    ball_state=packet.ball_state,
                    possession_result=packet.possession_result,
                    camera_motion=packet.camera_motion if self.enable_cmc else None,
                    cut_result=packet.cut_result if self.enable_reid else None,
                    reid_count=self.reid.total_reassignments if self.reid else 0,
                    active_event=packet.active_event,
                    ball_trail=self.ball_tracker.trail if self.ball_tracker else None,
                    fps=cur_fps,
                    frame_idx=frame_idx + 1,
                    total_frames=total_frames,
                    device_name="RTX 4050 (CUDA)" if "cuda" in self.detector.device else "CPU",
                    xg_scores=packet.xg_scores,
                )

                # 2D Tactical Radar Overlay
                if self.tactical_radar is not None and len(packet.player_positions_m) > 0:
                    player_team_colors = None
                    if packet.team_result is not None:
                        player_indices = np.where(detections.class_ids == 0)[0]
                        player_team_colors = [
                            packet.team_result.team_colors[idx]
                            for idx in player_indices
                            if idx < len(packet.team_result.team_colors)
                        ]

                    radar_img = self.tactical_radar.render_radar(
                        player_positions_m=packet.player_positions_m,
                        player_track_ids=players_res.tracker_ids,
                        ball_position_m=packet.ball_pos_m,
                        team_colors=player_team_colors,
                        tactical_spatial_result=packet.tactical_spatial_result,
                        possession_result=packet.possession_result,
                    )
                    annotated_frame = self.tactical_radar.overlay_on_frame(annotated_frame, radar_img)

                # Write to disk if configured
                if writer is not None:
                    writer.write(annotated_frame)

                # Format live telemetry JSON packet for WebSocket / Live Stream clients
                telemetry = self._extract_telemetry_dict(
                    packet=packet,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    fps=cur_fps,
                    elapsed_time=elapsed_total,
                )

                # Trigger real-time frame streaming callback
                if self.on_frame_processed is not None:
                    try:
                        self.on_frame_processed(frame_idx, telemetry, annotated_frame)
                    except Exception:
                        pass

                # Trigger progress callback
                if self.on_progress is not None and (frame_idx % 5 == 0 or frame_idx == total_frames - 1):
                    try:
                        self.on_progress(frame_idx + 1, total_frames, cur_fps)
                    except Exception:
                        pass

                t_egress = (time.perf_counter() - t0) * 1000.0
                self.stage_latencies["egress_ms"] = 0.9 * self.stage_latencies["egress_ms"] + 0.1 * t_egress

            if writer is not None:
                writer.release()

            # Finalize summary results
            poss_summary = self.ball_tracker.get_possession_summary() if self.ball_tracker else {}
            events_summary = self.event_detector.get_summary() if self.event_detector else None
            speed_summary = self.speed_estimator.get_team_summary() if self.speed_estimator else {}
            adv_summary = self.advanced_engine.get_summary() if self.advanced_engine else None

            if adv_summary is not None and self.passing_visualizer is not None and output_path:
                try:
                    diagram_dir = Path(output_path).parent / "tactics"
                    self.passing_visualizer.save_summary_diagrams(adv_summary, diagram_dir)
                except Exception:
                    pass

            self._results = {
                "total_frames": total_frames,
                "processed_frames": processed_count,
                "elapsed_time_s": round(time.time() - start_time, 2),
                "average_fps": cur_fps,
                "stage_latencies_ms": {k: round(v, 2) for k, v in self.stage_latencies.items()},
                "unique_players_tracked": len(unique_track_ids),
                "camera_cuts": len(cut_frames),
                "cut_frame_indices": cut_frames,
                "pan_distribution": pan_counts,
                "possession": poss_summary,
                "tactical_space": {
                    "avg_team_a_control_pct": round(float(np.mean(team_a_control_list)), 1) if team_a_control_list else 50.0,
                    "avg_team_b_control_pct": round(float(np.mean(team_b_control_list)), 1) if team_b_control_list else 50.0,
                },
                "events": events_summary.to_dict() if events_summary else {},
                "speed_kinematics": speed_summary,
                "advanced_tactics": adv_summary.to_dict() if adv_summary else {},
                "output_video": output_path,
            }

        except Exception as e:
            self._exceptions.append(e)
            self._stop_event.set()

    def _extract_telemetry_dict(
        self,
        packet: AnalyticsPacket,
        frame_idx: int,
        total_frames: int,
        fps: float,
        elapsed_time: float,
    ) -> Dict[str, Any]:
        """Extracts serializable tactical telemetry for WebSocket / Live Stream clients."""
        players_res = packet.detections.get_players()
        players_list = []
        for i in range(len(players_res.xyxy)):
            box = players_res.xyxy[i].tolist()
            tid = int(players_res.tracker_ids[i]) if (players_res.tracker_ids is not None and i < len(players_res.tracker_ids)) else -1
            pos_m = packet.player_positions_m[i].tolist() if i < len(packet.player_positions_m) else None
            
            speed_kmh = 0.0
            dist_m = 0.0
            speed_cat = "STANDING"
            if tid in packet.player_metrics:
                pm = packet.player_metrics[tid]
                speed_kmh = round(pm.current_speed_kmh, 1)
                dist_m = round(pm.cumulative_distance_m, 1)
                speed_cat = pm.speed_category

            team_id = 0
            team_name = "Team A"
            team_color = [255, 50, 50]
            if packet.team_result is not None and i < len(packet.team_result.team_ids):
                team_id = int(packet.team_result.team_ids[i])
                if isinstance(packet.team_result.team_names, list) and i < len(packet.team_result.team_names):
                    team_name = packet.team_result.team_names[i]
                elif isinstance(packet.team_result.team_names, dict):
                    team_name = packet.team_result.team_names.get(team_id, f"Team {team_id}")
                else:
                    team_name = f"Team {team_id}"

                if i < len(packet.team_result.team_colors):
                    team_color = [int(c) for c in packet.team_result.team_colors[i]]

            players_list.append({
                "id": tid,
                "bbox": [round(x, 1) for x in box],
                "pos_m": [round(p, 2) for p in pos_m] if pos_m else None,
                "team_id": team_id,
                "team_name": team_name,
                "team_color": team_color,
                "speed_kmh": speed_kmh,
                "distance_m": dist_m,
                "speed_category": speed_cat,
            })

        ball_data = {
            "detected": packet.ball_state is not None and packet.ball_state.position_m is not None,
            "pos_m": [round(p, 2) for p in packet.ball_pos_m] if packet.ball_pos_m else None,
            "speed_kmh": round(packet.ball_state.speed_kmh, 1) if packet.ball_state else 0.0,
            "carrier_id": packet.possession_result.carrier_track_id if (packet.possession_result and hasattr(packet.possession_result, 'carrier_track_id')) else (packet.possession_result.possessing_player_id if packet.possession_result else None),
            "possession_team": packet.possession_result.possessing_team_id if packet.possession_result else None,
            "is_contested": packet.possession_result.is_contested if packet.possession_result else False,
        }

        tactics_data = {
            "team_a_control_pct": packet.tactical_spatial_result.team_a_control_pct if packet.tactical_spatial_result else 50.0,
            "team_b_control_pct": packet.tactical_spatial_result.team_b_control_pct if packet.tactical_spatial_result else 50.0,
            "possession_team_a_pct": packet.possession_result.team_a_possession_pct if packet.possession_result else 50.0,
            "possession_team_b_pct": packet.possession_result.team_b_possession_pct if packet.possession_result else 50.0,
            "turnovers": packet.possession_result.turnover_count if packet.possession_result else 0,
            "xg_team_a": round(self.advanced_engine.total_xg_team_a, 2) if self.advanced_engine else 0.0,
            "xg_team_b": round(self.advanced_engine.total_xg_team_b, 2) if self.advanced_engine else 0.0,
            "xt_team_a": round(self.advanced_engine.total_xt_team_a, 2) if self.advanced_engine else 0.0,
            "xt_team_b": round(self.advanced_engine.total_xt_team_b, 2) if self.advanced_engine else 0.0,
        }

        camera_data = {
            "pan_direction": packet.camera_motion.pan_direction if packet.camera_motion else "STATIC",
            "pan_speed": round(abs(packet.camera_motion.dx_pixels), 1) if packet.camera_motion else 0.0,
            "zoom_factor": round(packet.camera_motion.zoom_factor, 2) if packet.camera_motion else 1.0,
            "is_cut": packet.cut_result.is_cut if packet.cut_result else False,
        }

        event_data = None
        if packet.active_event is not None:
            event_data = {
                "type": packet.active_event.event_type,
                "team": packet.active_event.team_name,
                "primary_player_id": packet.active_event.primary_player_id,
                "secondary_player_id": packet.active_event.secondary_player_id,
                "speed_kmh": round(packet.active_event.speed_kmh, 1),
                "distance_m": round(packet.active_event.distance_m, 1),
                "description": packet.active_event.description,
            }


        return {
            "frame_idx": frame_idx,
            "total_frames": total_frames,
            "fps": fps,
            "elapsed_time_s": round(elapsed_time, 2),
            "players": players_list,
            "ball": ball_data,
            "tactics": tactics_data,
            "camera": camera_data,
            "event": event_data,
        }

    # -------------------------------------------------------------------------
    # Pipeline Orchestration & Public API
    # -------------------------------------------------------------------------
    def start(self, source_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Starts the 4 pipeline threads asynchronously without blocking.
        Returns video metadata dictionary.
        """
        props = get_video_properties(str(source_path))
        preferred_device = get_device(self.options.get("device", self.config.get("system", {}).get("device", "auto")))

        # Initialize all model modules
        self._init_modules(props, preferred_device)

        self._stop_event.clear()
        self._exceptions.clear()

        # Thread 1: Ingest
        t_ingest = threading.Thread(
            target=self._ingest_worker,
            args=(source_path,),
            name="Stage1-IngestWorker",
            daemon=True,
        )

        # Thread 2: Inference & Tracking
        t_inference = threading.Thread(
            target=self._inference_worker,
            name="Stage2-InferenceWorker",
            daemon=True,
        )

        # Thread 3: Analytics
        t_analytics = threading.Thread(
            target=self._analytics_worker,
            args=(props["height"], props["width"]),
            name="Stage3-AnalyticsWorker",
            daemon=True,
        )

        # Thread 4: Egress & Live Streaming
        t_egress = threading.Thread(
            target=self._egress_worker,
            args=(output_path, props["fps"], props["width"], props["height"], props["frame_count"]),
            name="Stage4-EgressWorker",
            daemon=True,
        )

        self._threads = [t_ingest, t_inference, t_analytics, t_egress]
        for t in self._threads:
            t.start()

        return props

    def join(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Waits for all 4 pipeline stages to complete processing and returns match results.
        """
        for t in self._threads:
            t.join(timeout=timeout)

        if self._exceptions:
            raise self._exceptions[0]

        return self._results

    def stop(self):
        """Signals all worker threads to terminate immediately."""
        self._stop_event.set()
        # Drain queues to unblock any waiting puts
        for q in (self._ingest_queue, self._inference_queue, self._analytics_queue):
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass

    def run(self, source_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper: Starts the pipeline, joins on all threads, and returns match summary.
        """
        self.start(source_path=source_path, output_path=output_path)
        return self.join()
