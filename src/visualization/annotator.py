"""
Visual Annotation and HUD Overlay Module for Football Analysis (Polished & Hardened).
Supports Rounded Bounding Boxes, Persistent Tracking IDs, Fading Motion Trails,
Live Metric Speed (km/h), Role Badges [A]/[B]/[A-GK]/[B-GK]/[REF]/[COACH],
Tactical Spatial Dominance %, Compactness (m²), and Top Telemetry HUD.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


# Rich tactical color palette for unique player tracks (BGR format)
TRACK_PALETTE = [
    (0, 215, 255),    # Gold / Amber
    (255, 144, 30),   # Electric Blue
    (50, 205, 50),    # Lime Green
    (238, 130, 238),  # Violet
    (0, 165, 255),    # Orange
    (255, 191, 0),    # Deep Sky Blue
    (204, 50, 153),   # Medium Violet Red
    (0, 255, 255),    # Yellow
    (180, 105, 255),  # Hot Pink
    (128, 255, 0),    # Chartreuse
    (255, 99, 71),    # Light Steel Blue
    (0, 250, 154),    # Medium Spring Green
]


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Deterministically assign a vibrant color to each track ID."""
    if track_id < 0:
        return (0, 60, 255)  # Ball (Red)
    return TRACK_PALETTE[track_id % len(TRACK_PALETTE)]


class VideoAnnotator:
    """
    Renders high-quality bounding boxes, persistent tracking IDs, movement trails,
    metric speed indicators (km/h), role badges, pitch lines, and tactical HUD.
    """

    def __init__(
        self,
        player_color: Tuple[int, int, int] = (0, 215, 255),    # Bright Amber/Gold in BGR
        ball_color: Tuple[int, int, int] = (0, 60, 255),       # Bright Red/Orange in BGR
        referee_color: Tuple[int, int, int] = (0, 215, 255),   # Amber/Gold
        pitch_line_color: Tuple[int, int, int] = (0, 255, 255),# Cyan/Yellow
        box_thickness: int = 2,
        font_scale: float = 0.50,
        draw_conf: bool = True,
        draw_hud: bool = True,
        draw_tracks: bool = True,
        draw_trails: bool = True,
        draw_speed: bool = True,
        draw_team: bool = True,
        draw_pitch_boundary: bool = False,
        draw_pitch_lines: bool = False,
        unique_track_colors: bool = False,
    ):
        self.player_color = player_color
        self.ball_color = ball_color
        self.referee_color = referee_color
        self.pitch_line_color = pitch_line_color
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.draw_conf = draw_conf
        self.enable_hud = draw_hud
        self.enable_tracks = draw_tracks
        self.enable_trails = draw_trails
        self.draw_speed = draw_speed
        self.draw_team = draw_team
        self.draw_pitch_boundary = draw_pitch_boundary
        self.draw_pitch_lines = draw_pitch_lines
        self.unique_track_colors = unique_track_colors

    def _draw_rounded_box(
        self,
        image: np.ndarray,
        box: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int = 2,
        radius: int = 6,
    ):
        """Draw bounding box with smooth rounded edges."""
        x1, y1, x2, y2 = map(int, box)
        h, w = y2 - y1, x2 - x1
        r = min(radius, max(1, h // 4), max(1, w // 4))

        cv2.line(image, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)

        cv2.ellipse(image, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

    def _draw_badge(
        self,
        image: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """Draw a pill badge with anti-aliased label text."""
        x, y = pos
        (tw, th), baseline = cv2.getTextSize(text, self.font, self.font_scale, 1)
        padding = 4

        bx1, by1 = x, y - th - padding * 2
        bx2, by2 = x + tw + padding * 2, y

        if by1 < 0:
            by1 = y
            by2 = y + th + padding * 2

        sub_img = image[max(0, by1):min(image.shape[0], by2), max(0, bx1):min(image.shape[1], bx2)]
        if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
            overlay = sub_img.copy()
            overlay[:] = bg_color
            cv2.addWeighted(overlay, 0.85, sub_img, 0.15, 0, sub_img)

        cv2.rectangle(image, (bx1, by1), (bx2, by2), bg_color, 1, cv2.LINE_AA)
        text_y = by2 - padding - baseline // 2

        luminance = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
        actual_text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)

        cv2.putText(image, text, (bx1 + padding, text_y), self.font, self.font_scale, actual_text_color, 1, cv2.LINE_AA)

    def draw_pitch(
        self,
        frame: np.ndarray,
        pitch_result: Any,
    ) -> np.ndarray:
        """
        Render dynamic pitch boundary and detected field markings.
        """
        if pitch_result is None:
            return frame

        if self.draw_pitch_boundary and hasattr(pitch_result, "polygon") and len(pitch_result.polygon) > 0:
            cv2.polylines(
                frame,
                [pitch_result.polygon],
                isClosed=True,
                color=(0, 255, 128),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        if self.draw_pitch_lines and hasattr(pitch_result, "lines") and len(pitch_result.lines) > 0:
            for line in pitch_result.lines:
                x1, y1, x2, y2 = map(int, line)
                cv2.line(frame, (x1, y1), (x2, y2), self.pitch_line_color, 2, cv2.LINE_AA)

        return frame

    def draw_trails(
        self,
        frame: np.ndarray,
        trails: Dict[int, List[Tuple[float, float]]],
        team_result: Optional[Any] = None,
        tracker_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render smooth fading movement trajectories behind each player's feet.
        """
        for tid, points in trails.items():
            if len(points) < 2:
                continue

            color = get_track_color(tid)
            if team_result is not None and tracker_ids is not None:
                matches = np.where(tracker_ids == tid)[0]
                if len(matches) > 0:
                    color = team_result.team_colors[matches[0]]

            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
                progress = (i + 1) / len(points)
                thickness = max(1, int(2.5 * progress))
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

            latest_pt = (int(points[-1][0]), int(points[-1][1]))
            cv2.circle(frame, latest_pt, 3, color, -1, cv2.LINE_AA)

        return frame

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Any,
        team_result: Optional[Any] = None,
        player_metrics: Optional[Dict[int, Any]] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes, persistent track IDs, role badges, metric speeds, and class badges.
        """
        annotated = frame.copy()
        tracker_ids = getattr(detections, "tracker_ids", None)

        trails = getattr(detections, "trails", None)
        if self.enable_trails and trails:
            self.draw_trails(annotated, trails, team_result=team_result, tracker_ids=tracker_ids)

        num_items = len(detections.xyxy)

        for i in range(num_items):
            box = detections.xyxy[i]
            conf = detections.confidences[i]
            cid = detections.class_ids[i]
            cname = detections.class_names[i] if i < len(detections.class_names) else "object"
            tid = tracker_ids[i] if (tracker_ids is not None and i < len(tracker_ids)) else -1

            speed_str = ""
            if self.draw_speed and player_metrics and tid in player_metrics:
                speed_val = player_metrics[tid].current_speed_kmh
                if speed_val >= 2.0:
                    speed_str = f" {speed_val:.1f}km/h"

            role_prefix = ""
            if cid == 32:  # Ball
                color = self.ball_color
                label = f"Ball {conf:.2f}" if self.draw_conf else "Ball"
            elif cid == 0:  # Person (Player, GK, Ref, Coach)
                if team_result is not None and i < len(team_result.team_colors):
                    color = team_result.team_colors[i]
                    team_name = team_result.team_names[i]
                    if team_name == "Team A":
                        role_prefix = "[A] "
                    elif team_name == "Team B":
                        role_prefix = "[B] "
                    elif team_name == "Team A GK":
                        role_prefix = "[A-GK] "
                    elif team_name == "Team B GK":
                        role_prefix = "[B-GK] "
                    elif team_name == "Coach":
                        role_prefix = "[COACH] "
                    else:
                        role_prefix = "[REF] "
                elif self.unique_track_colors and tid >= 0:
                    color = get_track_color(int(tid))
                else:
                    color = self.player_color

                if self.enable_tracks and tid >= 0:
                    label = f"{role_prefix}#{tid}{speed_str}" if speed_str else f"{role_prefix}#{tid}"
                else:
                    label = f"{role_prefix}Player {conf:.2f}" if self.draw_conf else f"{role_prefix}Player"
            else:
                color = self.referee_color
                label = f"{cname} {conf:.2f}" if self.draw_conf else cname

            self._draw_rounded_box(annotated, box, color, self.box_thickness)
            self._draw_badge(annotated, label, (int(box[0]), int(box[1]) - 4), color)

            # Foot contact marker
            if cid == 0:
                foot_x = int((box[0] + box[2]) / 2.0)
                foot_y = int(box[3])
                axis_w = max(4, int((box[2] - box[0]) / 4.0))
                axis_h = max(2, int(axis_w / 2.5))
                cv2.ellipse(annotated, (foot_x, foot_y), (axis_w, axis_h), 0, 0, 360, color, 1, cv2.LINE_AA)

        return annotated

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        frame_idx: int,
        total_frames: int,
        num_players: int,
        ball_detected: bool,
        active_tracks: int = 0,
        pitch_detected: bool = False,
        team_counts: Optional[Dict[str, int]] = None,
        tactical_spatial_result: Optional[Any] = None,
        team_summary: Optional[Dict[str, float]] = None,
        device_name: str = "GPU",
    ) -> np.ndarray:
        """
        Draw a sleek tactical analytics HUD bar at the top of the video frame.
        """
        h, w, _ = frame.shape
        hud_height = 48

        top_bar = frame[0:hud_height, 0:w].copy()
        dark_overlay = np.full_like(top_bar, (18, 22, 28))
        cv2.addWeighted(dark_overlay, 0.78, top_bar, 0.22, 0, top_bar)
        frame[0:hud_height, 0:w] = top_bar

        cv2.line(frame, (0, hud_height), (w, hud_height), (0, 215, 255), 2, cv2.LINE_AA)

        max_speed = team_summary.get("max_speed_kmh", 0.0) if team_summary else 0.0
        tot_dist = team_summary.get("total_distance_km", 0.0) if team_summary else 0.0

        cnt_a = team_counts.get("Team A", 0) if team_counts else 0
        cnt_b = team_counts.get("Team B", 0) if team_counts else 0

        pct_a = tactical_spatial_result.team_a_control_pct if tactical_spatial_result else 50.0
        pct_b = tactical_spatial_result.team_b_control_pct if tactical_spatial_result else 50.0
        area_a = tactical_spatial_result.team_a_area_m2 if tactical_spatial_result else 0.0

        hud_items = [
            f"FRAME: {frame_idx:04d}/{total_frames:04d}",
            f"DEVICE: {device_name}",
            f"SPEED: {fps:.1f} FPS",
            f"TEAM A ({pct_a:.0f}%): {cnt_a:02d} | TEAM B ({pct_b:.0f}%): {cnt_b:02d}" if (cnt_a + cnt_b > 0) else f"PLAYERS: {num_players:02d}",
            f"COMPACT: {area_a:.0f} m²" if area_a > 0 else (f"MAX SPRINT: {max_speed:.1f} KM/H" if max_speed > 0 else "PITCH: LOCKED"),
            f"TOT DIST: {tot_dist:.2f} KM" if tot_dist > 0 else f"BALL: {'FOUND' if ball_detected else 'SEARCHING'}",
        ]

        section_width = w // len(hud_items)
        for idx, text in enumerate(hud_items):
            x = idx * section_width + 12
            y = 30
            if "BALL" in text or "SPRINT" in text:
                dot_color = (0, 255, 128)
                cv2.circle(frame, (x - 6, y - 6), 4, dot_color, -1, cv2.LINE_AA)
            elif "PITCH" in text or "COMPACT" in text:
                dot_color = (0, 255, 128) if pitch_detected else (80, 80, 200)
                cv2.circle(frame, (x - 6, y - 6), 4, dot_color, -1, cv2.LINE_AA)
            elif "TEAM" in text or "DIST" in text:
                cv2.circle(frame, (x - 6, y - 6), 4, (0, 215, 255), -1, cv2.LINE_AA)

            cv2.putText(
                frame,
                text,
                (x + 4, y),
                self.font,
                0.44,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        return frame

    def annotate(
        self,
        frame: np.ndarray,
        detections: Any,
        pitch_result: Optional[Any] = None,
        team_result: Optional[Any] = None,
        tactical_spatial_result: Optional[Any] = None,
        player_metrics: Optional[Dict[int, Any]] = None,
        team_summary: Optional[Dict[str, float]] = None,
        fps: float = 0.0,
        frame_idx: int = 0,
        total_frames: int = 0,
        device_name: str = "GPU",
    ) -> np.ndarray:
        """
        Complete annotation pipeline combining pitch lines, role badges, speed indicators, spatial metrics, and HUD.
        """
        annotated = self.draw_pitch(frame, pitch_result)
        annotated = self.draw_detections(annotated, detections, team_result=team_result, player_metrics=player_metrics)

        if self.enable_hud:
            num_players = len(detections.get_players().xyxy) if hasattr(detections, "get_players") else 0
            ball_detected = len(detections.get_ball().xyxy) > 0 if hasattr(detections, "get_ball") else False
            
            tracker_ids = getattr(detections, "tracker_ids", None)
            active_tracks = len(np.unique(tracker_ids[tracker_ids >= 0])) if tracker_ids is not None else 0
            pitch_detected = (
                pitch_result is not None
                and hasattr(pitch_result, "pitch_area_ratio")
                and pitch_result.pitch_area_ratio >= 0.20
            )

            annotated = self.draw_hud(
                annotated,
                fps=fps,
                frame_idx=frame_idx,
                total_frames=total_frames,
                num_players=num_players,
                ball_detected=ball_detected,
                active_tracks=active_tracks,
                pitch_detected=pitch_detected,
                team_counts=team_result.team_counts if team_result else None,
                tactical_spatial_result=tactical_spatial_result,
                team_summary=team_summary,
                device_name=device_name,
            )

        return annotated
