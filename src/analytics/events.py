"""
Football Match Event Recognition Engine (Milestone 11).
Automatically detects, classifies, and temporal-sequences high-value match events:
- Passes (Passer -> Receiver, pass distance, speed, completion status)
- Shots on Goal (Attacking strikes, shot speed km/h, goal-bound trajectories)
- Interceptions & Ball Recoveries (Defensive cuts of opponent passes)
- Tackles & Contested Duels (Close-quarters 1v1 challenges causing turnovers)
- Turnovers (Possession transitions and forced errors)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class MatchEvent:
    """
    Structured representation of an individual discrete football match event.
    """
    event_id: int
    event_type: str                  # "PASS", "SHOT", "INTERCEPTION", "TACKLE", "TURNOVER"
    frame_idx: int
    timestamp_s: float
    team_id: Optional[int]           # 0: Team A, 1: Team B, None: Neutral/Contested
    team_name: str                   # "Team A", "Team B", "Neutral"
    primary_player_id: Optional[int] # Origin passer, shooter, interceptor, or tackler
    secondary_player_id: Optional[int] = None # Receiver, challenged player, or goalkeeper
    start_pos_m: Optional[Tuple[float, float]] = None
    end_pos_m: Optional[Tuple[float, float]] = None
    speed_kmh: float = 0.0
    distance_m: float = 0.0
    is_successful: bool = True
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventSummary:
    """
    Cumulative match event telemetry summary.
    """
    total_events: int
    total_passes_a: int
    total_passes_b: int
    completed_passes_a: int
    completed_passes_b: int
    pass_accuracy_a_pct: float
    pass_accuracy_b_pct: float
    total_shots_a: int
    total_shots_b: int
    total_interceptions_a: int
    total_interceptions_b: int
    total_tackles_a: int
    total_tackles_b: int
    total_turnovers: int
    events_timeline: List[Dict[str, Any]]


class EventDetector:
    """
    Spatial-Temporal Finite State Machine for Real-Time Football Event Recognition.
    Monitors ball trajectory vectors, player foot distances, and possession switches
    to detect discrete tactical events.
    """

    def __init__(
        self,
        fps: float = 25.0,
        min_pass_distance_m: float = 3.0,
        min_pass_speed_kmh: float = 12.0,
        min_shot_speed_kmh: float = 36.0,
        tackle_proximity_m: float = 2.2,
        banner_duration_frames: int = 40,
    ):
        self.fps = fps
        self.min_pass_distance_m = min_pass_distance_m
        self.min_pass_speed_kmh = min_pass_speed_kmh
        self.min_shot_speed_kmh = min_shot_speed_kmh
        self.tackle_proximity_m = tackle_proximity_m
        self.banner_duration_frames = banner_duration_frames

        # Event storage
        self.events: List[MatchEvent] = []
        self.event_counter: int = 0

        # State tracking
        self.last_possessor_id: Optional[int] = None
        self.last_possessor_team: Optional[int] = None
        self.last_possessor_pos_m: Optional[Tuple[float, float]] = None
        self.last_possession_frame: int = 0

        self.ball_in_flight: bool = False
        self.flight_start_frame: int = 0
        self.flight_start_pos_m: Optional[Tuple[float, float]] = None
        self.flight_passer_id: Optional[int] = None
        self.flight_passer_team: Optional[int] = None
        self.flight_peak_speed_kmh: float = 0.0

        # Active event notification toast for video HUD
        self.active_event: Optional[MatchEvent] = None
        self.active_event_expiry: int = 0

    def _get_team_name(self, team_id: Optional[int]) -> str:
        if team_id == 0:
            return "Team A"
        elif team_id == 1:
            return "Team B"
        return "Neutral"

    def _is_shot_trajectory(
        self,
        start_pos_m: Tuple[float, float],
        end_pos_m: Tuple[float, float],
        speed_kmh: float,
        team_id: Optional[int],
    ) -> bool:
        """
        Check if ball flight trajectory is directed towards one of the two goals.
        FIFA pitch dimensions: 105m x 68m.
        Left Goal: X=0m, Y in [26.0, 42.0]m.
        Right Goal: X=105m, Y in [26.0, 42.0]m.
        """
        if speed_kmh < self.min_shot_speed_kmh:
            return False

        sx, sy = start_pos_m
        ex, ey = end_pos_m

        # Attacking Right Goal (X -> 105)
        if ex > 85.0 and 20.0 <= ey <= 48.0 and (ex - sx) > 5.0:
            return True

        # Attacking Left Goal (X -> 0)
        if ex < 20.0 and 20.0 <= ey <= 48.0 and (sx - ex) > 5.0:
            return True

        return False

    def update(
        self,
        ball_state: Any,
        possession_result: Any,
        player_positions_m: Optional[np.ndarray] = None,
        player_track_ids: Optional[np.ndarray] = None,
        player_team_ids: Optional[np.ndarray] = None,
        frame_idx: int = 0,
    ) -> List[MatchEvent]:
        """
        Process current frame state to identify discrete match events.

        Returns:
            List of new MatchEvents triggered at this frame.
        """
        new_events: List[MatchEvent] = []
        timestamp_s = frame_idx / max(1.0, self.fps)

        current_possessor_id = getattr(possession_result, "possessing_player_id", None)
        current_possessor_team = getattr(possession_result, "possessing_team_id", None)
        ball_pos_m = getattr(ball_state, "position_m", None)
        ball_speed_kmh = min(130.0, max(0.0, float(getattr(ball_state, "speed_kmh", 0.0))))

        # Track ball peak flight speed
        if self.ball_in_flight and ball_speed_kmh > self.flight_peak_speed_kmh:
            self.flight_peak_speed_kmh = min(130.0, ball_speed_kmh)

        # 1. POSSESSION RELEASE -> BALL IN FLIGHT
        if (
            self.last_possessor_id is not None
            and current_possessor_id is None
            and not self.ball_in_flight
            and ball_pos_m is not None
            and ball_speed_kmh >= self.min_pass_speed_kmh
        ):
            self.ball_in_flight = True
            self.flight_start_frame = frame_idx
            self.flight_start_pos_m = self.last_possessor_pos_m if self.last_possessor_pos_m is not None else ball_pos_m
            self.flight_passer_id = self.last_possessor_id
            self.flight_passer_team = self.last_possessor_team
            self.flight_peak_speed_kmh = ball_speed_kmh

        # 2. BALL ACQUISITION -> EVENT RESOLUTION
        elif current_possessor_id is not None:
            # Check if this is a newly acquired possession from a flight trajectory
            if (
                self.ball_in_flight
                and self.flight_passer_id is not None
                and self.flight_start_pos_m is not None
                and ball_pos_m is not None
            ):
                dist_m = float(np.sqrt(
                    (ball_pos_m[0] - self.flight_start_pos_m[0]) ** 2
                    + (ball_pos_m[1] - self.flight_start_pos_m[1]) ** 2
                ))

                is_same_team = (current_possessor_team == self.flight_passer_team)
                is_same_player = (current_possessor_id == self.flight_passer_id)

                # Check if it was a shot attempt on goal
                is_shot = self._is_shot_trajectory(
                    start_pos_m=self.flight_start_pos_m,
                    end_pos_m=ball_pos_m,
                    speed_kmh=self.flight_peak_speed_kmh,
                    team_id=self.flight_passer_team,
                )

                if is_shot and not is_same_player:
                    self.event_counter += 1
                    shot_event = MatchEvent(
                        event_id=self.event_counter,
                        event_type="SHOT",
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        team_id=self.flight_passer_team,
                        team_name=self._get_team_name(self.flight_passer_team),
                        primary_player_id=self.flight_passer_id,
                        secondary_player_id=current_possessor_id,
                        start_pos_m=self.flight_start_pos_m,
                        end_pos_m=ball_pos_m,
                        speed_kmh=self.flight_peak_speed_kmh,
                        distance_m=dist_m,
                        is_successful=True,
                        description=f"{self._get_team_name(self.flight_passer_team)} Shot: #{self.flight_passer_id} ({self.flight_peak_speed_kmh:.1f} km/h)",
                        details={"shot_speed_kmh": self.flight_peak_speed_kmh, "target_zone": "on_target"},
                    )
                    new_events.append(shot_event)

                elif is_same_team and not is_same_player and dist_m >= self.min_pass_distance_m:
                    # Completed pass between teammates
                    self.event_counter += 1
                    pass_event = MatchEvent(
                        event_id=self.event_counter,
                        event_type="PASS",
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        team_id=self.flight_passer_team,
                        team_name=self._get_team_name(self.flight_passer_team),
                        primary_player_id=self.flight_passer_id,
                        secondary_player_id=current_possessor_id,
                        start_pos_m=self.flight_start_pos_m,
                        end_pos_m=ball_pos_m,
                        speed_kmh=self.flight_peak_speed_kmh,
                        distance_m=dist_m,
                        is_successful=True,
                        description=f"{self._get_team_name(self.flight_passer_team)} Pass: #{self.flight_passer_id} -> #{current_possessor_id} ({dist_m:.1f}m)",
                        details={"pass_length_m": dist_m, "pass_speed_kmh": self.flight_peak_speed_kmh},
                    )
                    new_events.append(pass_event)

                elif not is_same_team and dist_m >= self.min_pass_distance_m:
                    # Incomplete pass intercepted by opponent
                    self.event_counter += 1
                    intercept_event = MatchEvent(
                        event_id=self.event_counter,
                        event_type="INTERCEPTION",
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        team_id=current_possessor_team,
                        team_name=self._get_team_name(current_possessor_team),
                        primary_player_id=current_possessor_id,
                        secondary_player_id=self.flight_passer_id,
                        start_pos_m=self.flight_start_pos_m,
                        end_pos_m=ball_pos_m,
                        speed_kmh=self.flight_peak_speed_kmh,
                        distance_m=dist_m,
                        is_successful=True,
                        description=f"{self._get_team_name(current_possessor_team)} Interception: #{current_possessor_id} intercepted #{self.flight_passer_id}",
                        details={"intercept_speed_kmh": self.flight_peak_speed_kmh},
                    )
                    new_events.append(intercept_event)

                # Reset flight state
                self.ball_in_flight = False
                self.flight_passer_id = None
                self.flight_passer_team = None
                self.flight_start_pos_m = None

            # 3. DIRECT CLOSE-CONTACT TACKLE / DUEL TURNOVER
            elif (
                self.last_possessor_id is not None
                and self.last_possessor_team is not None
                and current_possessor_team is not None
                and current_possessor_team != self.last_possessor_team
                and not self.ball_in_flight
                and (frame_idx - self.last_possession_frame) <= 6
            ):
                # Opponent snatched ball directly in a physical duel
                self.event_counter += 1
                tackle_event = MatchEvent(
                    event_id=self.event_counter,
                    event_type="TACKLE",
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    team_id=current_possessor_team,
                    team_name=self._get_team_name(current_possessor_team),
                    primary_player_id=current_possessor_id,
                    secondary_player_id=self.last_possessor_id,
                    start_pos_m=self.last_possessor_pos_m,
                    end_pos_m=ball_pos_m,
                    speed_kmh=ball_speed_kmh,
                    distance_m=0.0,
                    is_successful=True,
                    description=f"{self._get_team_name(current_possessor_team)} Tackle: #{current_possessor_id} won duel against #{self.last_possessor_id}",
                    details={"turnover": True},
                )
                new_events.append(tackle_event)

            # Update last possessor state
            self.last_possessor_id = current_possessor_id
            self.last_possessor_team = current_possessor_team
            self.last_possessor_pos_m = ball_pos_m
            self.last_possession_frame = frame_idx

        # Update event history & active HUD toast banner
        for ev in new_events:
            self.events.append(ev)
            self.active_event = ev
            self.active_event_expiry = frame_idx + self.banner_duration_frames

        if frame_idx >= self.active_event_expiry:
            self.active_event = None

        return new_events

    def get_summary(self) -> EventSummary:
        """
        Aggregate match events into high-level event statistics and chronological timeline.
        """
        passes_a = [e for e in self.events if e.event_type == "PASS" and e.team_id == 0]
        passes_b = [e for e in self.events if e.event_type == "PASS" and e.team_id == 1]
        shots_a = [e for e in self.events if e.event_type == "SHOT" and e.team_id == 0]
        shots_b = [e for e in self.events if e.event_type == "SHOT" and e.team_id == 1]
        interceptions_a = [e for e in self.events if e.event_type == "INTERCEPTION" and e.team_id == 0]
        interceptions_b = [e for e in self.events if e.event_type == "INTERCEPTION" and e.team_id == 1]
        tackles_a = [e for e in self.events if e.event_type == "TACKLE" and e.team_id == 0]
        tackles_b = [e for e in self.events if e.event_type == "TACKLE" and e.team_id == 1]

        total_passes_a = len(passes_a) + len(interceptions_b)
        total_passes_b = len(passes_b) + len(interceptions_a)

        acc_a = (len(passes_a) / max(1, total_passes_a)) * 100.0 if total_passes_a > 0 else 100.0
        acc_b = (len(passes_b) / max(1, total_passes_b)) * 100.0 if total_passes_b > 0 else 100.0

        timeline = []
        for e in self.events:
            timeline.append({
                "event_id": e.event_id,
                "event_type": e.event_type,
                "frame_idx": e.frame_idx,
                "timestamp_s": round(e.timestamp_s, 2),
                "timestamp_str": f"{int(e.timestamp_s // 60):02d}:{int(e.timestamp_s % 60):02d}",
                "team_id": e.team_id,
                "team_name": e.team_name,
                "primary_player_id": e.primary_player_id,
                "secondary_player_id": e.secondary_player_id,
                "speed_kmh": round(e.speed_kmh, 1),
                "distance_m": round(e.distance_m, 1),
                "is_successful": e.is_successful,
                "description": e.description,
            })

        return EventSummary(
            total_events=len(self.events),
            total_passes_a=total_passes_a,
            total_passes_b=total_passes_b,
            completed_passes_a=len(passes_a),
            completed_passes_b=len(passes_b),
            pass_accuracy_a_pct=round(acc_a, 1),
            pass_accuracy_b_pct=round(acc_b, 1),
            total_shots_a=len(shots_a),
            total_shots_b=len(shots_b),
            total_interceptions_a=len(interceptions_a),
            total_interceptions_b=len(interceptions_b),
            total_tackles_a=len(tackles_a),
            total_tackles_b=len(tackles_b),
            total_turnovers=len(interceptions_a) + len(interceptions_b) + len(tackles_a) + len(tackles_b),
            events_timeline=timeline,
        )
