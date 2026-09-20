"""
Unit Tests for Football Match Event Recognition Engine (Milestone 11).
"""

from types import SimpleNamespace
import numpy as np
import pytest

from src.analytics.events import EventDetector, MatchEvent, EventSummary


def test_event_detector_init():
    detector = EventDetector(fps=25.0)
    assert detector.fps == 25.0
    assert len(detector.events) == 0
    assert detector.active_event is None
    summary = detector.get_summary()
    assert summary.total_events == 0
    assert summary.pass_accuracy_a_pct == 100.0


def test_completed_pass_detection():
    detector = EventDetector(fps=25.0, min_pass_distance_m=3.0, min_pass_speed_kmh=12.0)

    # Frame 0: Player 10 (Team A / 0) has possession at (40.0, 30.0)
    ball_state = SimpleNamespace(position_m=(40.0, 30.0), speed_kmh=2.0)
    possession = SimpleNamespace(possessing_player_id=10, possessing_team_id=0)
    detector.update(ball_state, possession, frame_idx=0)
    assert detector.last_possessor_id == 10

    # Frame 1: Ball released, flying at 25 km/h
    ball_state = SimpleNamespace(position_m=(45.0, 30.0), speed_kmh=25.0)
    possession = SimpleNamespace(possessing_player_id=None, possessing_team_id=None)
    detector.update(ball_state, possession, frame_idx=1)
    assert detector.ball_in_flight is True
    assert detector.flight_passer_id == 10

    # Frame 10: Ball received by Player 7 (Team A / 0) at (55.0, 30.0) (15m distance)
    ball_state = SimpleNamespace(position_m=(55.0, 30.0), speed_kmh=8.0)
    possession = SimpleNamespace(possessing_player_id=7, possessing_team_id=0)
    new_events = detector.update(ball_state, possession, frame_idx=10)

    assert len(new_events) == 1
    ev = new_events[0]
    assert ev.event_type == "PASS"
    assert ev.primary_player_id == 10
    assert ev.secondary_player_id == 7
    assert ev.team_id == 0
    assert ev.team_name == "Team A"
    assert ev.distance_m == pytest.approx(15.0, 0.1)
    assert ev.speed_kmh >= 25.0
    assert detector.active_event == ev


def test_interception_detection():
    detector = EventDetector(fps=25.0)

    # Frame 0: Player 10 (Team A / 0) with ball
    ball_state = SimpleNamespace(position_m=(40.0, 30.0), speed_kmh=1.0)
    possession = SimpleNamespace(possessing_player_id=10, possessing_team_id=0)
    detector.update(ball_state, possession, frame_idx=0)

    # Frame 1: Ball in flight
    ball_state = SimpleNamespace(position_m=(45.0, 30.0), speed_kmh=22.0)
    possession = SimpleNamespace(possessing_player_id=None, possessing_team_id=None)
    detector.update(ball_state, possession, frame_idx=1)

    # Frame 8: Opponent Player 24 (Team B / 1) cuts the pass at (52.0, 30.0)
    ball_state = SimpleNamespace(position_m=(52.0, 30.0), speed_kmh=6.0)
    possession = SimpleNamespace(possessing_player_id=24, possessing_team_id=1)
    new_events = detector.update(ball_state, possession, frame_idx=8)

    assert len(new_events) == 1
    ev = new_events[0]
    assert ev.event_type == "INTERCEPTION"
    assert ev.primary_player_id == 24
    assert ev.secondary_player_id == 10
    assert ev.team_id == 1
    assert ev.team_name == "Team B"
    assert ev.distance_m == pytest.approx(12.0, 0.1)


def test_shot_on_goal_detection():
    detector = EventDetector(fps=25.0, min_shot_speed_kmh=36.0)

    # Frame 0: Player 9 (Team A / 0) at (75.0, 34.0)
    ball_state = SimpleNamespace(position_m=(75.0, 34.0), speed_kmh=5.0)
    possession = SimpleNamespace(possessing_player_id=9, possessing_team_id=0)
    detector.update(ball_state, possession, frame_idx=0)

    # Frame 1: Powerful shot at 55 km/h towards right goal
    ball_state = SimpleNamespace(position_m=(82.0, 34.0), speed_kmh=55.0)
    possession = SimpleNamespace(possessing_player_id=None, possessing_team_id=None)
    detector.update(ball_state, possession, frame_idx=1)

    # Frame 6: Goalkeeper (Player 1, Team B / 1) saves ball at (96.0, 34.0)
    ball_state = SimpleNamespace(position_m=(96.0, 34.0), speed_kmh=12.0)
    possession = SimpleNamespace(possessing_player_id=1, possessing_team_id=1)
    new_events = detector.update(ball_state, possession, frame_idx=6)

    assert len(new_events) == 1
    ev = new_events[0]
    assert ev.event_type == "SHOT"
    assert ev.primary_player_id == 9
    assert ev.secondary_player_id == 1
    assert ev.team_id == 0
    assert ev.speed_kmh == 55.0


def test_tackle_duel_turnover():
    detector = EventDetector(fps=25.0)

    # Frame 0: Player 11 (Team A / 0) in possession
    ball_state = SimpleNamespace(position_m=(50.0, 30.0), speed_kmh=4.0)
    possession = SimpleNamespace(possessing_player_id=11, possessing_team_id=0)
    detector.update(ball_state, possession, frame_idx=0)

    # Frame 2: Direct duel turnover - Player 4 (Team B / 1) takes ball directly
    ball_state = SimpleNamespace(position_m=(50.2, 30.1), speed_kmh=5.0)
    possession = SimpleNamespace(possessing_player_id=4, possessing_team_id=1)
    new_events = detector.update(ball_state, possession, frame_idx=2)

    assert len(new_events) == 1
    ev = new_events[0]
    assert ev.event_type == "TACKLE"
    assert ev.primary_player_id == 4
    assert ev.secondary_player_id == 11
    assert ev.team_id == 1
    assert ev.team_name == "Team B"


def test_dribble_same_player_no_pass():
    detector = EventDetector(fps=25.0)

    # Frame 0: Player 10 has possession
    ball_state = SimpleNamespace(position_m=(40.0, 30.0), speed_kmh=3.0)
    possession = SimpleNamespace(possessing_player_id=10, possessing_team_id=0)
    detector.update(ball_state, possession, frame_idx=0)

    # Frame 1: Released slightly
    ball_state = SimpleNamespace(position_m=(43.0, 30.0), speed_kmh=14.0)
    possession = SimpleNamespace(possessing_player_id=None, possessing_team_id=None)
    detector.update(ball_state, possession, frame_idx=1)

    # Frame 5: Same player touches ball again (dribble)
    ball_state = SimpleNamespace(position_m=(45.0, 30.0), speed_kmh=5.0)
    possession = SimpleNamespace(possessing_player_id=10, possessing_team_id=0)
    new_events = detector.update(ball_state, possession, frame_idx=5)

    assert len(new_events) == 0


def test_toast_banner_expiry():
    detector = EventDetector(fps=25.0, banner_duration_frames=10)

    # Trigger a pass at frame 5
    ball_state = SimpleNamespace(position_m=(40.0, 30.0), speed_kmh=2.0)
    possession = SimpleNamespace(possessing_player_id=10, possessing_team_id=0)
    detector.update(ball_state, possession, frame_idx=0)

    ball_state = SimpleNamespace(position_m=(45.0, 30.0), speed_kmh=20.0)
    possession = SimpleNamespace(possessing_player_id=None, possessing_team_id=None)
    detector.update(ball_state, possession, frame_idx=1)

    ball_state = SimpleNamespace(position_m=(55.0, 30.0), speed_kmh=5.0)
    possession = SimpleNamespace(possessing_player_id=7, possessing_team_id=0)
    detector.update(ball_state, possession, frame_idx=5)

    assert detector.active_event is not None
    assert detector.active_event.event_type == "PASS"

    # At frame 10 (within 10-frame banner duration): active_event remains
    detector.update(ball_state, possession, frame_idx=10)
    assert detector.active_event is not None

    # At frame 15 (>= 5 + 10): active_event expires
    detector.update(ball_state, possession, frame_idx=15)
    assert detector.active_event is None


def test_event_summary_aggregation():
    detector = EventDetector(fps=25.0)

    # 1. Pass Team A (10 -> 7)
    detector.update(SimpleNamespace(position_m=(40.0, 30.0), speed_kmh=2.0), SimpleNamespace(possessing_player_id=10, possessing_team_id=0), frame_idx=0)
    detector.update(SimpleNamespace(position_m=(45.0, 30.0), speed_kmh=20.0), SimpleNamespace(possessing_player_id=None, possessing_team_id=None), frame_idx=1)
    detector.update(SimpleNamespace(position_m=(55.0, 30.0), speed_kmh=5.0), SimpleNamespace(possessing_player_id=7, possessing_team_id=0), frame_idx=5)

    # 2. Interception by Team B (7 -> intercepted by 24)
    detector.update(SimpleNamespace(position_m=(58.0, 30.0), speed_kmh=22.0), SimpleNamespace(possessing_player_id=None, possessing_team_id=None), frame_idx=10)
    detector.update(SimpleNamespace(position_m=(68.0, 30.0), speed_kmh=5.0), SimpleNamespace(possessing_player_id=24, possessing_team_id=1), frame_idx=15)

    # 3. Tackle by Team A (24 tackled by 11)
    detector.update(SimpleNamespace(position_m=(68.2, 30.1), speed_kmh=4.0), SimpleNamespace(possessing_player_id=11, possessing_team_id=0), frame_idx=17)

    # 4. Shot by Team A (11 shoots on goal at 92.0, 34.0, keeper 1 saves)
    detector.update(SimpleNamespace(position_m=(75.0, 34.0), speed_kmh=48.0), SimpleNamespace(possessing_player_id=None, possessing_team_id=None), frame_idx=20)
    detector.update(SimpleNamespace(position_m=(92.0, 34.0), speed_kmh=5.0), SimpleNamespace(possessing_player_id=1, possessing_team_id=1), frame_idx=25)

    summary = detector.get_summary()

    assert summary.total_events == 4
    assert summary.completed_passes_a == 1
    assert summary.total_interceptions_b == 1
    assert summary.total_passes_a == 2  # 1 completed + 1 intercepted
    assert summary.pass_accuracy_a_pct == 50.0
    assert summary.total_tackles_a == 1
    assert summary.total_shots_a == 1
    assert summary.total_turnovers == 2  # 1 interception + 1 tackle
    assert len(summary.events_timeline) == 4
    assert summary.events_timeline[0]["event_type"] == "PASS"
    assert summary.events_timeline[1]["event_type"] == "INTERCEPTION"
    assert summary.events_timeline[2]["event_type"] == "TACKLE"
    assert summary.events_timeline[3]["event_type"] == "SHOT"
