"""
Advanced Tactical Football Analytics: Expected Goals (xG), Expected Threat (xT),
and Passing Network / Formation Topology Engine (Milestone 17).
"""

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ShotEvent:
    """Telemetry record for an individual shot on goal."""
    shot_id: int
    frame_idx: int
    timestamp_s: float
    shooter_id: int
    team_id: int
    team_name: str
    shot_pos_m: Tuple[float, float]
    target_goal_m: Tuple[float, float]
    distance_to_goal_m: float
    visible_angle_deg: float
    defenders_in_cone: int
    shot_speed_kmh: float
    xg_value: float
    is_goal: bool = False
    is_on_target: bool = True
    description: str = ""


@dataclass
class ThreatAction:
    """Telemetry record for a progressive pass or carry creating Expected Threat (xT)."""
    action_id: int
    frame_idx: int
    timestamp_s: float
    player_id: int
    team_id: int
    team_name: str
    action_type: str  # "PASS" or "CARRY"
    start_pos_m: Tuple[float, float]
    end_pos_m: Tuple[float, float]
    xt_start: float
    xt_end: float
    delta_xt: float
    is_progressive: bool
    description: str = ""


@dataclass
class PassingNode:
    """Player node in the tactical passing network."""
    player_id: int
    team_id: int
    team_name: str
    avg_x_m: float
    avg_y_m: float
    total_passes_made: int
    total_passes_received: int
    total_involvements: int
    xt_created: float
    role_label: str = "MF"


@dataclass
class PassingLink:
    """Directed connection between two passing teammates."""
    passer_id: int
    receiver_id: int
    team_id: int
    pass_count: int
    total_distance_m: float
    avg_speed_kmh: float
    total_delta_xt: float


@dataclass
class PassingNetworkSummary:
    """Summary of a team's passing network and inferred formation topology."""
    team_id: int
    team_name: str
    formation_str: str  # e.g., "4-3-3", "4-2-3-1", "3-5-2"
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    centrality_leader_id: Optional[int]
    top_passing_pair: Optional[Tuple[int, int, int]]  # (p1, p2, count)
    network_density: float
    average_pass_length_m: float


@dataclass
class AdvancedTacticsSummary:
    """Cumulative match summary for xG, xT, and Passing Networks."""
    total_xg_team_a: float
    total_xg_team_b: float
    total_xt_team_a: float
    total_xt_team_b: float
    shots_count_team_a: int
    shots_count_team_b: int
    progressive_passes_a: int
    progressive_passes_b: int
    shots: List[Dict[str, Any]]
    threat_actions: List[Dict[str, Any]]
    passing_networks: Dict[str, Any]
    top_creators_a: List[Dict[str, Any]]
    top_creators_b: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        from dataclasses import asdict
        return asdict(self)


# =============================================================================
# 1. Expected Goals (xG) Calibrated Spatial Model
# =============================================================================
class ExpectedGoalsModel:
    """
    Calibrated spatial-geometric Expected Goals (xG) Model.
    Computes probabilistic shot conversion rate [0.0, 1.0] from shot distance,
    visible goalmouth angle, defender congestion, and shot strike speed.
    """

    GOAL_WIDTH_M = 7.32
    GOAL_LEFT = (0.0, 34.0)
    GOAL_RIGHT = (105.0, 34.0)

    def __init__(self):
        # Calibrated logistic regression weights from broadcast match event distributions
        self.b0 = 0.85      # Baseline intercept
        self.b_dist = -0.115 # Distance penalty per meter
        self.b_angle = 0.042 # Visual angle bonus per degree
        self.b_def = -0.280  # Defender in shooting cone penalty
        self.b_speed = 0.012 # Ball speed strike bonus

    def evaluate_shot(
        self,
        shot_pos_m: Tuple[float, float],
        defenders_m: Optional[np.ndarray] = None,
        shot_speed_kmh: float = 65.0,
        attacking_direction: int = 1,  # 1: Attacking Right (105m), -1: Attacking Left (0m)
    ) -> Tuple[float, float, float, int]:
        """
        Calculates xG value, distance to goal center, visible angle, and defender count.

        Returns:
            Tuple of (xg_value, distance_m, angle_deg, defenders_in_cone)
        """
        x, y = float(shot_pos_m[0]), float(shot_pos_m[1])
        goal_x = 105.0 if attacking_direction >= 0 else 0.0
        goal_y = 34.0

        # 1. Euclidean distance to center of goal
        dist = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)
        dist = max(0.5, dist)

        # 2. Visible Goalmouth Angle
        post_top = (goal_x, 34.0 - self.GOAL_WIDTH_M / 2.0)
        post_bot = (goal_x, 34.0 + self.GOAL_WIDTH_M / 2.0)

        v1 = (post_top[0] - x, post_top[1] - y)
        v2 = (post_bot[0] - x, post_bot[1] - y)

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        cos_angle = max(-1.0, min(1.0, dot / max(1e-5, mag1 * mag2)))
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        # 3. Defender Congestion in Shooting Cone
        defenders_in_cone = 0
        if defenders_m is not None and len(defenders_m) > 0:
            for def_pos in defenders_m:
                dx, dy = float(def_pos[0]), float(def_pos[1])
                # Check if defender is in front of the shooter towards the goal
                if (attacking_direction >= 0 and x <= dx <= goal_x) or (attacking_direction < 0 and goal_x <= dx <= x):
                    # Check if defender is within lateral cone width
                    t = abs(dx - x) / max(0.1, abs(goal_x - x))
                    cone_half_width = (self.GOAL_WIDTH_M / 2.0 + 1.5) * t + 0.5 * (1.0 - t)
                    if abs(dy - goal_y) <= cone_half_width + 1.2:
                        defenders_in_cone += 1

        # 4. Logistic Sigmoid xG Estimation
        logit = (
            self.b0
            + self.b_dist * dist
            + self.b_angle * angle_deg
            + self.b_def * min(4, defenders_in_cone)
            + self.b_speed * (max(20.0, min(120.0, shot_speed_kmh)) - 60.0) / 10.0
        )
        xg = 1.0 / (1.0 + math.exp(-logit))
        xg = max(0.01, min(0.96, xg))

        return round(xg, 3), round(dist, 1), round(angle_deg, 1), defenders_in_cone


# =============================================================================
# 2. Expected Threat (xT) 16x12 Markov Pitch Grid Model
# =============================================================================
class ExpectedThreatModel:
    """
    Positional Expected Threat (xT) Markov Transition Model.
    Discretizes the 105m x 68m football pitch into a 16x12 spatial zone grid.
    Quantifies value added by progressive passes and carries towards the goal.
    """

    GRID_W = 16
    GRID_H = 12

    def __init__(self):
        # Base 16x12 positional threat surface (attacking Left -> Right towards column 15)
        # Higher values near penalty box center and Zone 14
        self.xt_grid = self._create_base_xt_grid()

    def _create_base_xt_grid(self) -> np.ndarray:
        grid = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
        center_y = (self.GRID_H - 1) / 2.0

        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                # Normalized distance to right goalmouth (col 15, center_y)
                dist_norm = math.sqrt((15.0 - x) ** 2 + 1.8 * (center_y - y) ** 2)
                # Exponential decay threat surface
                val = 0.28 * math.exp(-dist_norm / 4.8) + 0.006 * (x / 15.0) ** 1.5 + 0.003
                grid[y, x] = val

        # Boost central penalty area & Zone 14 (cols 11-14, rows 3-8)
        grid[3:9, 11:15] *= 1.35
        grid[4:8, 13:16] *= 1.45
        return np.clip(grid, 0.003, 0.35)

    def get_zone(self, pos_m: Tuple[float, float], attacking_direction: int = 1) -> Tuple[int, int]:
        """Maps continuous pitch coordinates (x, y) into (grid_x, grid_y) zone."""
        x, y = float(pos_m[0]), float(pos_m[1])
        x = max(0.0, min(104.9, x))
        y = max(0.0, min(67.9, y))

        # Mirror x if attacking left to right
        if attacking_direction < 0:
            x = 105.0 - x

        gx = int((x / 105.0) * self.GRID_W)
        gy = int((y / 68.0) * self.GRID_H)

        gx = max(0, min(self.GRID_W - 1, gx))
        gy = max(0, min(self.GRID_H - 1, gy))
        return gx, gy

    def get_xt_value(self, pos_m: Tuple[float, float], attacking_direction: int = 1) -> float:
        """Looks up the Expected Threat value at metric pitch position (x, y)."""
        gx, gy = self.get_zone(pos_m, attacking_direction)
        return float(self.xt_grid[gy, gx])

    def evaluate_action(
        self,
        start_pos_m: Tuple[float, float],
        end_pos_m: Tuple[float, float],
        action_type: str = "PASS",
        attacking_direction: int = 1,
    ) -> Tuple[float, float, float, bool]:
        """
        Calculates xt_start, xt_end, delta_xt, and progressive action classification.

        Returns:
            Tuple of (xt_start, xt_end, delta_xt, is_progressive)
        """
        xt_start = self.get_xt_value(start_pos_m, attacking_direction)
        xt_end = self.get_xt_value(end_pos_m, attacking_direction)
        delta_xt = xt_end - xt_start

        # Progressive action definition:
        # 1. Delta xT >= +0.012, OR
        # 2. Moves ball >= 25% closer to goal in opponent half (x >= 52.5m)
        goal_x = 105.0 if attacking_direction >= 0 else 0.0
        d_start = abs(goal_x - start_pos_m[0])
        d_end = abs(goal_x - end_pos_m[0])
        dist_gained = d_start - d_end

        is_prog = (delta_xt >= 0.012) or (dist_gained >= 8.0 and (d_start <= 55.0 or dist_gained >= 12.0))

        return round(xt_start, 4), round(xt_end, 4), round(delta_xt, 4), is_prog


# =============================================================================
# 3. Passing Network & Formation Topology Engine
# =============================================================================
class PassingNetworkEngine:
    """
    Accumulates directed player passing interactions, average on-pitch positions,
    graph centrality metrics, and infers team tactical formation (e.g. 4-3-3, 4-2-3-1).
    """

    def __init__(self):
        # Track ID -> list of (x, y) coordinates
        self.player_positions = defaultdict(list)
        # (Passer_ID, Receiver_ID) -> pass count
        self.pass_counts = defaultdict(int)
        # (Passer_ID, Receiver_ID) -> list of distances
        self.pass_distances = defaultdict(list)
        # (Passer_ID, Receiver_ID) -> list of speeds
        self.pass_speeds = defaultdict(list)
        # (Passer_ID, Receiver_ID) -> list of delta_xt
        self.pass_delta_xts = defaultdict(list)

        # Player ID -> total passes made / received
        self.player_passes_made = defaultdict(int)
        self.player_passes_received = defaultdict(int)
        self.player_xt_created = defaultdict(float)
        self.player_teams = {}

    def add_player_position(self, player_id: int, team_id: int, pos_m: Tuple[float, float]):
        """Records on-pitch position for computing mean tactical coordinates."""
        if player_id < 0:
            return
        self.player_positions[player_id].append((float(pos_m[0]), float(pos_m[1])))
        self.player_teams[player_id] = team_id

    def add_pass(
        self,
        passer_id: int,
        receiver_id: int,
        team_id: int,
        distance_m: float,
        speed_kmh: float,
        delta_xt: float = 0.0,
    ):
        """Records a completed pass between two teammates."""
        if passer_id < 0 or receiver_id < 0 or passer_id == receiver_id:
            return

        self.pass_counts[(passer_id, receiver_id)] += 1
        self.pass_distances[(passer_id, receiver_id)].append(distance_m)
        self.pass_speeds[(passer_id, receiver_id)].append(speed_kmh)
        self.pass_delta_xts[(passer_id, receiver_id)].append(delta_xt)

        self.player_passes_made[passer_id] += 1
        self.player_passes_received[receiver_id] += 1
        self.player_xt_created[passer_id] += max(0.0, delta_xt)
        self.player_teams[passer_id] = team_id
        self.player_teams[receiver_id] = team_id

    def infer_formation(self, team_nodes: List[Dict[str, Any]], team_id: int) -> str:
        """
        Infers tactical formation structure (e.g. 4-3-3, 4-2-3-1, 3-5-2)
        by clustering players along the longitudinal pitch axis (X-coordinates).
        """
        if len(team_nodes) < 6:
            return "4-3-3 (Standard)"

        # Sort outfield players by X coordinate
        outfield = [n for n in team_nodes if n.get("role_label") != "GK"]
        if not outfield:
            outfield = team_nodes

        # Sort by longitudinal distance (defenders -> midfielders -> attackers)
        # Determine team attacking direction by mean X
        mean_x = np.mean([n["avg_x_m"] for n in outfield])
        attacking_right = mean_x <= 55.0  # If defenders have smaller X

        sorted_players = sorted(
            outfield,
            key=lambda p: p["avg_x_m"] if attacking_right else -p["avg_x_m"],
        )

        n = len(sorted_players)
        if n >= 10:
            # 10 outfield players: Split into Defense, Midfield, Attack lines
            # Cluster by X-distance gaps
            xs = [p["avg_x_m"] if attacking_right else (105.0 - p["avg_x_m"]) for p in sorted_players]
            
            # Simple heuristic distribution
            def_count = 4
            mid_count = 3
            att_count = 3

            # Check for 3-back or 5-back
            d1 = xs[2] - xs[0]
            d2 = xs[3] - xs[0]
            if d2 > 12.0:
                def_count = 3
                mid_count = 4
                att_count = n - 7
            elif n >= 10 and xs[4] - xs[0] < 8.0:
                def_count = 5
                mid_count = 3
                att_count = 2

            return f"{def_count}-{mid_count}-{max(1, att_count)}"
        else:
            return "4-3-3"

    def get_team_network(self, team_id: int, min_passes_per_link: int = 1) -> PassingNetworkSummary:
        """Generates passing network graph summary for a specific team."""
        team_name = "Team A" if team_id == 0 else "Team B"

        # 1. Compute Node Averages
        nodes = []
        team_player_ids = [pid for pid, tid in self.player_teams.items() if tid == team_id]

        for pid in team_player_ids:
            pos_list = self.player_positions.get(pid, [])
            if len(pos_list) == 0:
                avg_x, avg_y = 52.5, 34.0
            else:
                arr = np.array(pos_list)
                avg_x = float(np.mean(arr[:, 0]))
                avg_y = float(np.mean(arr[:, 1]))

            p_made = self.player_passes_made.get(pid, 0)
            p_rec = self.player_passes_received.get(pid, 0)
            involvements = p_made + p_rec
            xt_val = round(self.player_xt_created.get(pid, 0.0), 3)

            # Assign preliminary tactical role
            role = "MF"
            if avg_x < 22.0 or avg_x > 83.0:
                role = "GK" if (26.0 <= avg_y <= 42.0 and (avg_x < 14.0 or avg_x > 91.0)) else "DF"
            elif avg_x < 42.0 or avg_x > 63.0:
                role = "DF"
            elif avg_x > 70.0 or avg_x < 35.0:
                role = "FW"

            nodes.append({
                "player_id": pid,
                "team_id": team_id,
                "team_name": team_name,
                "avg_x_m": round(avg_x, 1),
                "avg_y_m": round(avg_y, 1),
                "total_passes_made": p_made,
                "total_passes_received": p_rec,
                "total_involvements": involvements,
                "xt_created": xt_val,
                "role_label": role,
            })

        # 2. Compute Links
        links = []
        all_dists = []
        top_pair = None
        max_pair_count = 0

        for (p1, p2), count in self.pass_counts.items():
            if self.player_teams.get(p1) == team_id and self.player_teams.get(p2) == team_id:
                if count >= min_passes_per_link:
                    dists = self.pass_distances.get((p1, p2), [15.0])
                    speeds = self.pass_speeds.get((p1, p2), [40.0])
                    delta_xts = self.pass_delta_xts.get((p1, p2), [0.0])

                    links.append({
                        "passer_id": p1,
                        "receiver_id": p2,
                        "team_id": team_id,
                        "pass_count": count,
                        "avg_distance_m": round(float(np.mean(dists)), 1),
                        "avg_speed_kmh": round(float(np.mean(speeds)), 1),
                        "total_delta_xt": round(float(np.sum(delta_xts)), 3),
                    })
                    all_dists.extend(dists)

                    if count > max_pair_count:
                        max_pair_count = count
                        top_pair = (p1, p2, count)

        # 3. Network Metrics
        formation = self.infer_formation(nodes, team_id)
        centrality_leader = max(nodes, key=lambda n: n["total_involvements"])["player_id"] if nodes else None
        density = round(len(links) / max(1, len(nodes) * (len(nodes) - 1)), 3) if len(nodes) > 1 else 0.0
        avg_dist = round(float(np.mean(all_dists)), 1) if all_dists else 14.5

        return PassingNetworkSummary(
            team_id=team_id,
            team_name=team_name,
            formation_str=formation,
            nodes=nodes,
            links=links,
            centrality_leader_id=centrality_leader,
            top_passing_pair=top_pair,
            network_density=density,
            average_pass_length_m=avg_dist,
        )


# =============================================================================
# 4. Unified Advanced Tactical Intelligence Engine (Milestone 17)
# =============================================================================
class TacticalAdvancedEngine:
    """
    High-Level Unified Tactical Engine coordinating Expected Goals (xG),
    Expected Threat (xT), and Passing Network Topology.
    """

    def __init__(self):
        self.xg_model = ExpectedGoalsModel()
        self.xt_model = ExpectedThreatModel()
        self.passing_network = PassingNetworkEngine()

        self.shots: List[ShotEvent] = []
        self.threat_actions: List[ThreatAction] = []
        self.shot_counter = 0
        self.action_counter = 0

        self.total_xg_team_a = 0.0
        self.total_xg_team_b = 0.0
        self.total_xt_team_a = 0.0
        self.total_xt_team_b = 0.0

    def update_player_positions(
        self,
        player_track_ids: Optional[np.ndarray],
        player_team_ids: Optional[np.ndarray],
        player_positions_m: Optional[np.ndarray],
    ):
        """Accumulates player track positions for passing network mean coordinates."""
        if (
            player_track_ids is None
            or player_team_ids is None
            or player_positions_m is None
            or len(player_track_ids) == 0
            or len(player_track_ids) != len(player_positions_m)
            or len(player_track_ids) != len(player_team_ids)
        ):
            return

        for tid, team_id, pos in zip(player_track_ids, player_team_ids, player_positions_m):
            if tid >= 0 and team_id in (0, 1):
                self.passing_network.add_player_position(int(tid), int(team_id), (float(pos[0]), float(pos[1])))

    def record_shot_event(
        self,
        shooter_id: int,
        team_id: int,
        shot_pos_m: Tuple[float, float],
        frame_idx: int,
        timestamp_s: float,
        shot_speed_kmh: float = 65.0,
        defenders_m: Optional[np.ndarray] = None,
        is_goal: bool = False,
        attacking_direction: Optional[int] = None,
    ) -> ShotEvent:
        """Evaluates and records an attacking shot on goal with calibrated xG value."""
        self.shot_counter += 1
        team_name = "Team A" if team_id == 0 else "Team B"

        if attacking_direction is None:
            # Infer attacking direction based on team: Team A attacks Right (105m), Team B attacks Left (0m)
            attacking_direction = 1 if team_id == 0 else -1

        goal_target = (105.0, 34.0) if attacking_direction >= 0 else (0.0, 34.0)

        xg, dist, angle, def_count = self.xg_model.evaluate_shot(
            shot_pos_m=shot_pos_m,
            defenders_m=defenders_m,
            shot_speed_kmh=shot_speed_kmh,
            attacking_direction=attacking_direction,
        )

        if team_id == 0:
            self.total_xg_team_a += xg
        elif team_id == 1:
            self.total_xg_team_b += xg

        shot = ShotEvent(
            shot_id=self.shot_counter,
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            shooter_id=shooter_id,
            team_id=team_id,
            team_name=team_name,
            shot_pos_m=shot_pos_m,
            target_goal_m=goal_target,
            distance_to_goal_m=dist,
            visible_angle_deg=angle,
            defenders_in_cone=def_count,
            shot_speed_kmh=shot_speed_kmh,
            xg_value=xg,
            is_goal=is_goal,
            description=f"Shot by Player #{shooter_id} ({xg} xG, {dist}m, {shot_speed_kmh:.1f} km/h)",
        )
        self.shots.append(shot)
        return shot

    def record_pass_action(
        self,
        passer_id: int,
        receiver_id: int,
        team_id: int,
        start_pos_m: Tuple[float, float],
        end_pos_m: Tuple[float, float],
        frame_idx: int,
        timestamp_s: float,
        speed_kmh: float = 40.0,
        attacking_direction: Optional[int] = None,
    ) -> ThreatAction:
        """Evaluates and records a completed pass with Expected Threat (xT) gain and passing graph update."""
        self.action_counter += 1
        team_name = "Team A" if team_id == 0 else "Team B"

        if attacking_direction is None:
            attacking_direction = 1 if team_id == 0 else -1

        dist = math.sqrt((end_pos_m[0] - start_pos_m[0]) ** 2 + (end_pos_m[1] - start_pos_m[1]) ** 2)

        xt_start, xt_end, delta_xt, is_prog = self.xt_model.evaluate_action(
            start_pos_m=start_pos_m,
            end_pos_m=end_pos_m,
            action_type="PASS",
            attacking_direction=attacking_direction,
        )

        if delta_xt > 0:
            if team_id == 0:
                self.total_xt_team_a += delta_xt
            elif team_id == 1:
                self.total_xt_team_b += delta_xt

        # Update Passing Network Graph
        self.passing_network.add_pass(
            passer_id=passer_id,
            receiver_id=receiver_id,
            team_id=team_id,
            distance_m=dist,
            speed_kmh=speed_kmh,
            delta_xt=delta_xt,
        )

        action = ThreatAction(
            action_id=self.action_counter,
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            player_id=passer_id,
            team_id=team_id,
            team_name=team_name,
            action_type="PASS",
            start_pos_m=start_pos_m,
            end_pos_m=end_pos_m,
            xt_start=xt_start,
            xt_end=xt_end,
            delta_xt=delta_xt,
            is_progressive=is_prog,
            description=f"Pass #{passer_id} -> #{receiver_id} (+{delta_xt:.3f} xT, {dist:.1f}m)",
        )
        self.threat_actions.append(action)
        return action

    def get_summary(self) -> AdvancedTacticsSummary:
        """Compiles comprehensive match summary of xG, xT, and Passing Networks."""
        net_a = self.passing_network.get_team_network(team_id=0)
        net_b = self.passing_network.get_team_network(team_id=1)

        prog_a = len([a for a in self.threat_actions if a.team_id == 0 and a.is_progressive])
        prog_b = len([a for a in self.threat_actions if a.team_id == 1 and a.is_progressive])

        shots_data = [
            {
                "shot_id": s.shot_id,
                "frame_idx": s.frame_idx,
                "timestamp_s": round(s.timestamp_s, 2),
                "shooter_id": s.shooter_id,
                "team_id": s.team_id,
                "team_name": s.team_name,
                "shot_pos_m": [round(s.shot_pos_m[0], 1), round(s.shot_pos_m[1], 1)],
                "distance_m": s.distance_to_goal_m,
                "angle_deg": s.visible_angle_deg,
                "xg_value": s.xg_value,
                "speed_kmh": round(s.shot_speed_kmh, 1),
                "is_goal": s.is_goal,
                "description": s.description,
            }
            for s in self.shots
        ]

        threat_data = [
            {
                "action_id": a.action_id,
                "frame_idx": a.frame_idx,
                "timestamp_s": round(a.timestamp_s, 2),
                "player_id": a.player_id,
                "team_id": a.team_id,
                "team_name": a.team_name,
                "action_type": a.action_type,
                "start_pos_m": [round(a.start_pos_m[0], 1), round(a.start_pos_m[1], 1)],
                "end_pos_m": [round(a.end_pos_m[0], 1), round(a.end_pos_m[1], 1)],
                "delta_xt": a.delta_xt,
                "is_progressive": a.is_progressive,
                "description": a.description,
            }
            for a in self.threat_actions[-20:]  # Latest 20 threat actions
        ]

        # Top Threat Creators
        top_a = sorted(net_a.nodes, key=lambda n: n["xt_created"], reverse=True)[:5]
        top_b = sorted(net_b.nodes, key=lambda n: n["xt_created"], reverse=True)[:5]

        return AdvancedTacticsSummary(
            total_xg_team_a=round(self.total_xg_team_a, 2),
            total_xg_team_b=round(self.total_xg_team_b, 2),
            total_xt_team_a=round(self.total_xt_team_a, 2),
            total_xt_team_b=round(self.total_xt_team_b, 2),
            shots_count_team_a=len([s for s in self.shots if s.team_id == 0]),
            shots_count_team_b=len([s for s in self.shots if s.team_id == 1]),
            progressive_passes_a=prog_a,
            progressive_passes_b=prog_b,
            shots=shots_data,
            threat_actions=threat_data,
            passing_networks={
                "team_a": {
                    "formation": net_a.formation_str,
                    "nodes": net_a.nodes,
                    "links": net_a.links,
                    "density": net_a.network_density,
                    "avg_pass_length_m": net_a.average_pass_length_m,
                    "centrality_leader": net_a.centrality_leader_id,
                },
                "team_b": {
                    "formation": net_b.formation_str,
                    "nodes": net_b.nodes,
                    "links": net_b.links,
                    "density": net_b.network_density,
                    "avg_pass_length_m": net_b.average_pass_length_m,
                    "centrality_leader": net_b.centrality_leader_id,
                },
            },
            top_creators_a=top_a,
            top_creators_b=top_b,
        )
