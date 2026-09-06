"""
Visual Annotation and HUD Overlay Module for Football Analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from ..detection.detector import DetectionResult


class VideoAnnotator:
    """
    Renders high-quality bounding boxes, labels, and real-time analytical HUD overlays.
    """

    def __init__(
        self,
        player_color: Tuple[int, int, int] = (0, 215, 255),    # Bright Amber/Gold in BGR
        ball_color: Tuple[int, int, int] = (0, 60, 255),       # Bright Red/Orange in BGR
        referee_color: Tuple[int, int, int] = (0, 255, 128),   # Neon Green
        team_colors: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((255, 60, 60), (60, 150, 255)), # Red and Blue
        box_thickness: int = 2,
        font_scale: float = 0.55,
        draw_conf: bool = True,
        draw_hud: bool = True,
    ):
        self.player_color = player_color
        self.ball_color = ball_color
        self.referee_color = referee_color
        self.team_colors = team_colors
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.draw_conf = draw_conf
        self.enable_hud = draw_hud

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
        r = min(radius, h // 4, w // 4)

        # Main lines
        cv2.line(image, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)

        # Corners
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

        # Ensure inside frame boundary
        if by1 < 0:
            by1 = y
            by2 = y + th + padding * 2

        # Semi-transparent badge background
        sub_img = image[max(0, by1):min(image.shape[0], by2), max(0, bx1):min(image.shape[1], bx2)]
        if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
            overlay = sub_img.copy()
            overlay[:] = bg_color
            cv2.addWeighted(overlay, 0.85, sub_img, 0.15, 0, sub_img)

        # Border for badge
        cv2.rectangle(image, (bx1, by1), (bx2, by2), bg_color, 1, cv2.LINE_AA)
        text_y = by2 - padding - baseline // 2
        cv2.putText(image, text, (bx1 + padding, text_y), self.font, self.font_scale, text_color, 1, cv2.LINE_AA)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: DetectionResult,
        team_ids: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and class badges on the frame.

        Args:
            frame: Video frame (H, W, 3) in BGR.
            detections: DetectionResult containing bounding boxes and confidences.
            team_ids: Dictionary mapping track_id to team_id.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        for i in range(detections.num_detections):
            box = detections.xyxy[i]
            conf = detections.confidences[i]
            cid = detections.class_ids[i]
            cname = detections.class_names[i]
            track_id = detections.track_ids[i] if detections.track_ids is not None else None

            # Choose color based on class
            if cid == 32:  # Ball
                color = self.ball_color
                label = f"Ball {conf:.2f}" if self.draw_conf else "Ball"
            elif cid == 0:  # Player/Person
                color = self.player_color
                label = f"Player {conf:.2f}" if self.draw_conf else "Player"
                
                # Update color and label if tracked and team assigned
                if track_id is not None:
                    label = f"P{track_id}"
                    if team_ids and track_id in team_ids:
                        t_id = team_ids[track_id]
                        if t_id >= 0 and t_id < len(self.team_colors):
                            color = self.team_colors[t_id]
                            label = f"T{t_id + 1} | {label}"
            else:
                color = self.referee_color
                label = f"{cname} {conf:.2f}" if self.draw_conf else cname
                if track_id is not None:
                    label = f"ID:{track_id} | {label}"

            # Draw box
            self._draw_rounded_box(annotated, box, color, self.box_thickness)

            # Draw badge above box
            self._draw_badge(annotated, label, (int(box[0]), int(box[1]) - 4), color)

        return annotated

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        frame_idx: int,
        total_frames: int,
        num_players: int,
        ball_detected: bool,
        device_name: str = "GPU",
    ) -> np.ndarray:
        """
        Draw a sleek tactical analytics HUD bar at the top of the video frame.
        """
        h, w, _ = frame.shape
        hud_height = 48

        # Create semi-transparent top bar
        top_bar = frame[0:hud_height, 0:w].copy()
        dark_overlay = np.full_like(top_bar, (18, 22, 28))
        cv2.addWeighted(dark_overlay, 0.78, top_bar, 0.22, 0, top_bar)
        frame[0:hud_height, 0:w] = top_bar

        # Bottom accent line (Cyan gradient / line)
        cv2.line(frame, (0, hud_height), (w, hud_height), (0, 215, 255), 2, cv2.LINE_AA)

        # HUD Text items
        hud_items = [
            f"FRAME: {frame_idx:04d}/{total_frames:04d}",
            f"DEVICE: {device_name}",
            f"SPEED: {fps:.1f} FPS",
            f"PLAYERS: {num_players:02d}",
            f"BALL: {'FOUND' if ball_detected else 'SEARCHING'}",
        ]

        section_width = w // len(hud_items)
        for idx, text in enumerate(hud_items):
            x = idx * section_width + 16
            y = 30
            # Indicator dot for ball status or live processing
            if "BALL" in text:
                dot_color = (0, 255, 128) if ball_detected else (80, 80, 200)
                cv2.circle(frame, (x - 6, y - 6), 4, dot_color, -1, cv2.LINE_AA)
            elif "PLAYERS" in text:
                cv2.circle(frame, (x - 6, y - 6), 4, (0, 215, 255), -1, cv2.LINE_AA)

            cv2.putText(
                frame,
                text,
                (x + 4, y),
                self.font,
                0.48,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        return frame

    def annotate(
        self,
        frame: np.ndarray,
        detections: DetectionResult,
        fps: float = 0.0,
        frame_idx: int = 0,
        total_frames: int = 0,
        device_name: str = "GPU",
        team_ids: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """
        Complete annotation pipeline combining detections and HUD.
        """
        annotated = self.draw_detections(frame, detections, team_ids=team_ids)

        if self.enable_hud:
            num_players = len(detections.get_players().xyxy)
            ball_detected = len(detections.get_ball().xyxy) > 0
            annotated = self.draw_hud(
                annotated,
                fps=fps,
                frame_idx=frame_idx,
                total_frames=total_frames,
                num_players=num_players,
                ball_detected=ball_detected,
                device_name=device_name,
            )

        return annotated

    def draw_minimap(
        self,
        frame: np.ndarray,
        detections: "DetectionResult",
        team_ids: Optional[Dict[int, int]] = None,
        map_width: int = 200,
        map_height: int = 130,
        corner: str = "bottom_right",
        alpha: float = 0.80,
    ) -> np.ndarray:
        """
        Draw a top-down 2D pitch minimap overlay with player positions.

        Each player is rendered as a small circle projected from their bounding
        box foot-point onto a scaled top-down pitch rectangle.

        Args:
            frame: Full video frame (H, W, 3) in BGR.
            detections: DetectionResult for the current frame.
            team_ids: Optional dict mapping track_id -> team_id for team colors.
            map_width: Width of the minimap in pixels.
            map_height: Height of the minimap in pixels.
            corner: Where to place the map: 'bottom_right', 'bottom_left',
                    'top_right', 'top_left'.
            alpha: Blend opacity of the minimap overlay (0.0–1.0).

        Returns:
            Frame with minimap overlay drawn in-place copy.
        """
        h_frame, w_frame = frame.shape[:2]
        annotated = frame.copy()

        # Determine minimap canvas position
        margin = 12
        if corner == "bottom_right":
            ox, oy = w_frame - map_width - margin, h_frame - map_height - margin
        elif corner == "bottom_left":
            ox, oy = margin, h_frame - map_height - margin
        elif corner == "top_right":
            ox, oy = w_frame - map_width - margin, margin
        else:  # top_left
            ox, oy = margin, margin

        # Draw pitch background
        minimap = np.zeros((map_height, map_width, 3), dtype=np.uint8)
        minimap[:] = (30, 100, 30)  # Dark green pitch
        # Pitch border
        cv2.rectangle(minimap, (2, 2), (map_width - 3, map_height - 3), (255, 255, 255), 1, cv2.LINE_AA)
        # Centre line
        cv2.line(minimap, (map_width // 2, 2), (map_width // 2, map_height - 3), (255, 255, 255), 1, cv2.LINE_AA)
        # Centre circle
        cv2.circle(minimap, (map_width // 2, map_height // 2), map_height // 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Project detections
        players = detections.get_players()
        for i in range(players.num_detections):
            box = players.xyxy[i]
            # Foot-point: bottom-centre of bbox
            foot_x = (box[0] + box[2]) / 2.0
            foot_y = box[3]

            # Scale to minimap coordinates
            mx = int((foot_x / w_frame) * map_width)
            my = int((foot_y / h_frame) * map_height)
            mx = max(4, min(map_width - 5, mx))
            my = max(4, min(map_height - 5, my))

            # Choose colour by team
            dot_color = self.player_color
            if team_ids is not None and players.track_ids is not None:
                tid = int(players.track_ids[i])
                t = team_ids.get(tid, -1)
                if 0 <= t < len(self.team_colors):
                    dot_color = self.team_colors[t]

            cv2.circle(minimap, (mx, my), 4, dot_color, -1, cv2.LINE_AA)
            cv2.circle(minimap, (mx, my), 4, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend minimap onto frame
        roi = annotated[oy : oy + map_height, ox : ox + map_width]
        if roi.shape[:2] == minimap.shape[:2]:
            cv2.addWeighted(minimap, alpha, roi, 1.0 - alpha, 0, roi)
            annotated[oy : oy + map_height, ox : ox + map_width] = roi

        return annotated

