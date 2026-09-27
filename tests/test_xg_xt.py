"""
Automated Unit Test Suite for Milestone 17:
Expected Goals (xG), Expected Threat (xT), and Passing Network / Formation Topology Engine.
"""

import numpy as np
import pytest

from src.analytics.xg_xt import (
    ExpectedGoalsModel,
    ExpectedThreatModel,
    PassingNetworkEngine,
    TacticalAdvancedEngine,
    ShotEvent,
    ThreatAction,
    PassingNetworkSummary,
    AdvancedTacticsSummary,
)
from src.visualization.passing_network import PassingNetworkVisualizer


class TestExpectedGoalsModel:
    """Tests for spatial geometric Expected Goals (xG) calculation."""

    def test_xg_distance_decay(self):
        """Verify that shot conversion probability strictly decreases with distance to goal."""
        model = ExpectedGoalsModel()
        # Central shots at 6m, 16m (edge of box), and 30m (long range)
        close_shot = model.evaluate_shot(shot_pos_m=(99.0, 34.0), attacking_direction=1)
        mid_shot = model.evaluate_shot(shot_pos_m=(89.0, 34.0), attacking_direction=1)
        far_shot = model.evaluate_shot(shot_pos_m=(75.0, 34.0), attacking_direction=1)

        xg_close, d_close, _, _ = close_shot
        xg_mid, d_mid, _, _ = mid_shot
        xg_far, d_far, _, _ = far_shot

        assert xg_close > xg_mid > xg_far
        assert d_close < d_mid < d_far
        assert 0.0 < xg_far < xg_close <= 1.0

    def test_xg_angle_monotonicity(self):
        """Verify that wider visual goal angles yield higher xG than narrow acute angles."""
        model = ExpectedGoalsModel()
        # Shot at 12m out centrally vs 12m out near the corner/byline
        central_shot = model.evaluate_shot(shot_pos_m=(93.0, 34.0), attacking_direction=1)
        acute_shot = model.evaluate_shot(shot_pos_m=(93.0, 10.0), attacking_direction=1)

        xg_central, _, angle_central, _ = central_shot
        xg_acute, _, angle_acute, _ = acute_shot

        assert angle_central > angle_acute
        assert xg_central > xg_acute

    def test_xg_defender_congestion_penalty(self):
        """Verify that defenders in the shooting cone reduce the xG value."""
        model = ExpectedGoalsModel()
        shot_pos = (90.0, 34.0)

        # 0 defenders
        open_shot = model.evaluate_shot(shot_pos_m=shot_pos, defenders_m=None, attacking_direction=1)
        
        # 2 defenders blocking the goal line
        defenders = np.array([
            [95.0, 33.5],
            [96.0, 34.5],
        ], dtype=np.float32)
        blocked_shot = model.evaluate_shot(shot_pos_m=shot_pos, defenders_m=defenders, attacking_direction=1)

        xg_open, _, _, def_count_open = open_shot
        xg_blocked, _, _, def_count_blocked = blocked_shot

        assert def_count_open == 0
        assert def_count_blocked >= 1
        assert xg_open > xg_blocked


class TestExpectedThreatModel:
    """Tests for 16x12 Expected Threat (xT) pitch grid and progressive action classification."""

    def test_xt_grid_spatial_gradient(self):
        """Verify that xT values increase towards opponent penalty area and Zone 14."""
        model = ExpectedThreatModel()
        xt_defensive_third = model.get_xt_value((20.0, 34.0), attacking_direction=1)
        xt_middle_third = model.get_xt_value((52.5, 34.0), attacking_direction=1)
        xt_zone_14 = model.get_xt_value((85.0, 34.0), attacking_direction=1)
        xt_six_yard_box = model.get_xt_value((100.0, 34.0), attacking_direction=1)

        assert xt_defensive_third < xt_middle_third < xt_zone_14 < xt_six_yard_box

    def test_xt_progressive_pass_action(self):
        """Verify forward passes advancing play into attacking third are marked progressive."""
        model = ExpectedThreatModel()
        # 20m forward pass into Zone 14
        xt_start, xt_end, delta_xt, is_prog = model.evaluate_action(
            start_pos_m=(60.0, 34.0),
            end_pos_m=(85.0, 34.0),
            action_type="PASS",
            attacking_direction=1,
        )

        assert delta_xt > 0
        assert xt_end > xt_start
        assert is_prog is True

    def test_xt_backward_pass_negative_delta(self):
        """Verify backward defensive passes reduce threat."""
        model = ExpectedThreatModel()
        xt_start, xt_end, delta_xt, is_prog = model.evaluate_action(
            start_pos_m=(80.0, 34.0),
            end_pos_m=(45.0, 34.0),
            action_type="PASS",
            attacking_direction=1,
        )

        assert delta_xt < 0
        assert is_prog is False


class TestPassingNetworkEngine:
    """Tests for Passing Network Graph, link accumulation, and formation inference."""

    def test_passing_network_aggregation(self):
        """Verify pass count, distances, and centrality aggregation."""
        engine = PassingNetworkEngine()
        
        # Add positions
        engine.add_player_position(10, team_id=0, pos_m=(50.0, 30.0))
        engine.add_player_position(10, team_id=0, pos_m=(52.0, 32.0))
        engine.add_player_position(7, team_id=0, pos_m=(70.0, 20.0))
        engine.add_player_position(9, team_id=0, pos_m=(85.0, 34.0))

        # Add passes
        engine.add_pass(passer_id=10, receiver_id=7, team_id=0, distance_m=22.0, speed_kmh=42.0, delta_xt=0.03)
        engine.add_pass(passer_id=10, receiver_id=7, team_id=0, distance_m=21.0, speed_kmh=44.0, delta_xt=0.02)
        engine.add_pass(passer_id=7, receiver_id=9, team_id=0, distance_m=18.0, speed_kmh=48.0, delta_xt=0.08)

        net_summary = engine.get_team_network(team_id=0)

        assert len(net_summary.nodes) == 3
        assert len(net_summary.links) == 2
        # Player #10 has 2 passes made, Player #7 has 2 received + 1 made = 3 involvements (leader)
        assert net_summary.centrality_leader_id == 7
        assert net_summary.top_passing_pair == (10, 7, 2)

    def test_formation_inference(self):
        """Verify formation clustering heuristic assigns valid formation strings."""
        engine = PassingNetworkEngine()
        nodes = [
            {"player_id": 1, "avg_x_m": 8.0, "avg_y_m": 34.0, "role_label": "GK", "total_involvements": 5},
            {"player_id": 2, "avg_x_m": 25.0, "avg_y_m": 12.0, "role_label": "DF", "total_involvements": 10},
            {"player_id": 3, "avg_x_m": 26.0, "avg_y_m": 26.0, "role_label": "DF", "total_involvements": 12},
            {"player_id": 4, "avg_x_m": 26.0, "avg_y_m": 42.0, "role_label": "DF", "total_involvements": 11},
            {"player_id": 5, "avg_x_m": 25.0, "avg_y_m": 56.0, "role_label": "DF", "total_involvements": 8},
            {"player_id": 6, "avg_x_m": 50.0, "avg_y_m": 20.0, "role_label": "MF", "total_involvements": 18},
            {"player_id": 7, "avg_x_m": 52.0, "avg_y_m": 34.0, "role_label": "MF", "total_involvements": 22},
            {"player_id": 8, "avg_x_m": 51.0, "avg_y_m": 48.0, "role_label": "MF", "total_involvements": 16},
            {"player_id": 9, "avg_x_m": 78.0, "avg_y_m": 16.0, "role_label": "FW", "total_involvements": 14},
            {"player_id": 10, "avg_x_m": 82.0, "avg_y_m": 34.0, "role_label": "FW", "total_involvements": 15},
            {"player_id": 11, "avg_x_m": 79.0, "avg_y_m": 52.0, "role_label": "FW", "total_involvements": 12},
        ]

        formation = engine.infer_formation(nodes, team_id=0)
        assert "-" in formation
        assert len(formation) >= 5


class TestTacticalAdvancedEngineAndVisualizer:
    """Tests for unified TacticalAdvancedEngine orchestrator and 2D visualizer."""

    def test_tactical_advanced_engine_summary(self):
        """Verify full engine workflow: position tracking, shot recording, pass actions, and summary dict."""
        engine = TacticalAdvancedEngine()

        # Feed positions
        track_ids = np.array([10, 19, 26], dtype=int)
        team_ids = np.array([0, 0, 1], dtype=int)
        positions = np.array([[60.0, 30.0], [80.0, 34.0], [30.0, 34.0]], dtype=np.float32)
        engine.update_player_positions(track_ids, team_ids, positions)

        # Record a pass
        engine.record_pass_action(
            passer_id=10,
            receiver_id=19,
            team_id=0,
            start_pos_m=(60.0, 30.0),
            end_pos_m=(80.0, 34.0),
            frame_idx=10,
            timestamp_s=0.4,
            speed_kmh=42.0,
        )

        # Record a shot
        engine.record_shot_event(
            shooter_id=19,
            team_id=0,
            shot_pos_m=(88.0, 34.0),
            frame_idx=25,
            timestamp_s=1.0,
            shot_speed_kmh=85.0,
        )

        summary = engine.get_summary()

        assert summary.total_xg_team_a > 0.0
        assert summary.total_xt_team_a > 0.0
        assert summary.shots_count_team_a == 1
        assert "team_a" in summary.passing_networks
        assert "team_b" in summary.passing_networks

        s_dict = summary.to_dict()
        assert isinstance(s_dict, dict)
        assert "total_xg_team_a" in s_dict

    def test_passing_network_visualizer_rendering(self):
        """Verify PassingNetworkVisualizer generates valid 2D top-down image canvases."""
        engine = TacticalAdvancedEngine()
        engine.passing_network.add_player_position(10, 0, (40.0, 30.0))
        engine.passing_network.add_player_position(19, 0, (75.0, 34.0))
        engine.passing_network.add_pass(10, 19, 0, distance_m=35.0, speed_kmh=50.0, delta_xt=0.06)

        visualizer = PassingNetworkVisualizer(canvas_width=600, canvas_height=400)
        net_a = engine.passing_network.get_team_network(team_id=0)

        img_team = visualizer.render_team_network(net_a, canvas_width=600, canvas_height=400)
        assert isinstance(img_team, np.ndarray)
        assert img_team.shape == (400, 600, 3)

        summary = engine.get_summary()
        img_dual = visualizer.render_dual_network(summary, canvas_width=1000, canvas_height=400)
        assert isinstance(img_dual, np.ndarray)
        assert img_dual.shape == (400, 1000, 3)
