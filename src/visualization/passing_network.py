"""
2D Tactical Passing Network and Formation Topology Visualizer (Milestone 17).
Renders weighted passing interaction links, player node centrality, average positional topology,
and tactical formation structures on top-down FIFA standard pitch canvases.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from ..calibration.template import PitchTemplate, PITCH_LENGTH_M, PITCH_WIDTH_M
from ..analytics.xg_xt import PassingNetworkSummary, AdvancedTacticsSummary


class PassingNetworkVisualizer:
    """
    Renders top-down 2D Passing Network and Formation Topology diagrams
    with weighted directed links, player involvement nodes, and formation telemetry.
    """

    def __init__(
        self,
        canvas_width: int = 800,
        canvas_height: int = 520,
        padding: int = 35,
        bg_color: Tuple[int, int, int] = (15, 20, 25),        # Deep tactical dark
        pitch_line_color: Tuple[int, int, int] = (180, 210, 190), # Pitch lines
        team_a_color: Tuple[int, int, int] = (255, 255, 255),  # White
        team_b_color: Tuple[int, int, int] = (50, 220, 100),   # Neon Green
    ):
        self.width = canvas_width
        self.height = canvas_height
        self.padding = padding
        self.bg_color = bg_color
        self.pitch_line_color = pitch_line_color
        self.team_a_color = team_a_color
        self.team_b_color = team_b_color
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.template = PitchTemplate()

    def _meters_to_pixels(self, x_m: float, y_m: float, w: int, h: int, pad: int) -> Tuple[int, int]:
        """Maps metric pitch coordinate (x, y) to pixel coordinate (px, py)."""
        field_w = w - 2 * pad
        field_h = h - 2 * pad - 50  # Reserve 50px top header for formation title

        sx = field_w / PITCH_LENGTH_M
        sy = field_h / PITCH_WIDTH_M

        cx = np.clip(x_m, 0.0, PITCH_LENGTH_M)
        cy = np.clip(y_m, 0.0, PITCH_WIDTH_M)

        px = int(pad + cx * sx)
        py = int(pad + 50 + (PITCH_WIDTH_M - cy) * sy)
        return (px, py)

    def render_team_network(
        self,
        network: PassingNetworkSummary,
        canvas_width: Optional[int] = None,
        canvas_height: Optional[int] = None,
        primary_color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Renders a single team's passing network and formation diagram.
        """
        w = canvas_width or self.width
        h = canvas_height or self.height
        pad = self.padding
        team_color = primary_color or (self.team_a_color if network.team_id == 0 else self.team_b_color)

        # 1. Base Pitch Canvas
        canvas = np.full((h, w, 3), self.bg_color, dtype=np.uint8)

        # Draw pitch markings
        field_x1, field_y1 = pad, pad + 50
        field_x2, field_y2 = w - pad, h - pad
        cv2.rectangle(canvas, (field_x1, field_y1), (field_x2, field_y2), self.pitch_line_color, 2, cv2.LINE_AA)

        # Halfway line & center circle
        mid_x = (field_x1 + field_x2) // 2
        mid_y = (field_y1 + field_y2) // 2
        cv2.line(canvas, (mid_x, field_y1), (mid_x, field_y2), self.pitch_line_color, 1, cv2.LINE_AA)
        circle_r = int((field_y2 - field_y1) * (9.15 / PITCH_WIDTH_M))
        cv2.circle(canvas, (mid_x, mid_y), max(10, circle_r), self.pitch_line_color, 1, cv2.LINE_AA)

        # Penalty boxes
        pen_w = int((field_x2 - field_x1) * (16.5 / PITCH_LENGTH_M))
        pen_h = int((field_y2 - field_y1) * (40.32 / PITCH_WIDTH_M))
        pen_top = mid_y - pen_h // 2
        pen_bot = mid_y + pen_h // 2
        # Left box
        cv2.rectangle(canvas, (field_x1, pen_top), (field_x1 + pen_w, pen_bot), self.pitch_line_color, 1, cv2.LINE_AA)
        # Right box
        cv2.rectangle(canvas, (field_x2 - pen_w, pen_top), (field_x2, pen_bot), self.pitch_line_color, 1, cv2.LINE_AA)

        # 2. Top Header & Formation Banner
        header_text = f"{network.team_name.upper()} - FORMATION: {network.formation_str}"
        cv2.putText(canvas, header_text, (pad, 30), self.font, 0.65, team_color, 2, cv2.LINE_AA)

        stats_text = (
            f"Density: {network.network_density:.2f} | "
            f"Avg Pass: {network.average_pass_length_m:.1f}m | "
            f"Hub: #{network.centrality_leader_id if network.centrality_leader_id is not None else 'N/A'}"
        )
        cv2.putText(canvas, stats_text, (pad, 48), self.font, 0.40, (180, 190, 200), 1, cv2.LINE_AA)

        # Build quick player position lookup
        node_coords: Dict[int, Tuple[int, int]] = {}
        for node in network.nodes:
            pid = node["player_id"]
            px, py = self._meters_to_pixels(node["avg_x_m"], node["avg_y_m"], w, h, pad)
            node_coords[pid] = (px, py)

        # 3. Draw Directed Passing Links
        for link in network.links:
            p1 = link["passer_id"]
            p2 = link["receiver_id"]
            count = link["pass_count"]
            if p1 not in node_coords or p2 not in node_coords:
                continue

            pt1 = node_coords[p1]
            pt2 = node_coords[p2]

            thickness = max(1, min(6, int(1 + count * 0.8)))
            link_alpha = min(0.85, 0.35 + count * 0.1)

            # Glowing link overlay
            link_color = (0, 220, 255) if count >= 3 else (140, 170, 190)
            overlay = canvas.copy()
            cv2.line(overlay, pt1, pt2, link_color, thickness, cv2.LINE_AA)
            cv2.addWeighted(overlay, link_alpha, canvas, 1.0 - link_alpha, 0, canvas)

        # 4. Draw Player Nodes
        for node in network.nodes:
            pid = node["player_id"]
            if pid not in node_coords:
                continue
            px, py = node_coords[pid]
            involvements = node["total_involvements"]
            node_r = max(10, min(22, 10 + int(involvements * 0.8)))

            # Node glow & fill
            cv2.circle(canvas, (px, py), node_r + 3, (0, 215, 255) if pid == network.centrality_leader_id else (40, 50, 60), 2, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), node_r, team_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), node_r, (20, 25, 30), 1, cv2.LINE_AA)

            # Node ID text
            id_str = f"{pid}"
            (tw, th), _ = cv2.getTextSize(id_str, self.font, 0.40, 1)
            text_color = (0, 0, 0) if (0.299 * team_color[2] + 0.587 * team_color[1] + 0.114 * team_color[0]) > 140 else (255, 255, 255)
            cv2.putText(canvas, id_str, (px - tw // 2, py + th // 2), self.font, 0.40, text_color, 1, cv2.LINE_AA)

            # Sub-label (Role & xT)
            role_label = f"{node.get('role_label', 'MF')}"
            (rtw, rth), _ = cv2.getTextSize(role_label, self.font, 0.32, 1)
            cv2.putText(canvas, role_label, (px - rtw // 2, py + node_r + 12), self.font, 0.32, (200, 210, 220), 1, cv2.LINE_AA)

        return canvas

    def render_dual_network(
        self,
        summary: AdvancedTacticsSummary,
        canvas_width: int = 1500,
        canvas_height: int = 560,
    ) -> np.ndarray:
        """
        Renders side-by-side comparative passing networks and formation structures for both teams.
        """
        net_a_data = summary.passing_networks.get("team_a", {})
        net_b_data = summary.passing_networks.get("team_b", {})

        net_a = PassingNetworkSummary(
            team_id=0,
            team_name="Team A",
            formation_str=net_a_data.get("formation", "4-3-3"),
            nodes=net_a_data.get("nodes", []),
            links=net_a_data.get("links", []),
            centrality_leader_id=net_a_data.get("centrality_leader"),
            top_passing_pair=None,
            network_density=net_a_data.get("density", 0.0),
            average_pass_length_m=net_a_data.get("avg_pass_length_m", 15.0),
        )

        net_b = PassingNetworkSummary(
            team_id=1,
            team_name="Team B",
            formation_str=net_b_data.get("formation", "4-3-3"),
            nodes=net_b_data.get("nodes", []),
            links=net_b_data.get("links", []),
            centrality_leader_id=net_b_data.get("centrality_leader"),
            top_passing_pair=None,
            network_density=net_b_data.get("density", 0.0),
            average_pass_length_m=net_b_data.get("avg_pass_length_m", 15.0),
        )

        single_w = canvas_width // 2 - 10
        img_a = self.render_team_network(net_a, canvas_width=single_w, canvas_height=canvas_height, primary_color=self.team_a_color)
        img_b = self.render_team_network(net_b, canvas_width=single_w, canvas_height=canvas_height, primary_color=self.team_b_color)

        combined = np.full((canvas_height, canvas_width, 3), (10, 14, 18), dtype=np.uint8)
        combined[:, :single_w] = img_a
        combined[:, single_w + 20:single_w + 20 + single_w] = img_b

        # Center separator line
        cv2.line(combined, (canvas_width // 2, 20), (canvas_width // 2, canvas_height - 20), (50, 60, 75), 1, cv2.LINE_AA)

        return combined

    def save_summary_diagrams(self, summary: AdvancedTacticsSummary, output_dir: Union[str, Path]):
        """Saves high-res passing network PNG diagrams to output directory."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dual_img = self.render_dual_network(summary)
        out_path = out_dir / "tactical_passing_networks.png"
        cv2.imwrite(str(out_path), dual_img)
        return str(out_path)
