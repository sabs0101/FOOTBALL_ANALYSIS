"""
Unit tests for Automated Team, Goalkeeper, Referee & Coach Identification.
"""

import numpy as np
import pytest
from src.team.classifier import TeamClassifier, TeamResult


def test_team_classifier_initialization():
    classifier = TeamClassifier(n_teams=2, history_window=15)
    assert classifier.n_teams == 2
    assert classifier.history_window == 15
    assert "gk_a" in classifier.palette
    assert "coach" in classifier.palette


def test_role_prediction_logic():
    """
    Test prediction on synthetic HSV profiles & spatial coordinates:
    - Outfield Team A (White)
    - Outfield Team B (Neon Green)
    - Team A Goalkeeper [A-GK] (Goal area X < 28m)
    - Referee [REF] (Central pitch X=50m, dark kit)
    - Coach [COACH] (Below touchline y2 > 770px)
    """
    classifier = TeamClassifier()

    # 1. White Outfield Jersey
    white_hsv = (80.0, 30.0, 200.0)
    assert classifier.predict_role(white_hsv, pos_m=(50.0, 34.0)) == 0  # Team A

    # 2. Neon Green Outfield Jersey
    green_hsv = (52.0, 130.0, 220.0)
    assert classifier.predict_role(green_hsv, pos_m=(60.0, 34.0)) == 1  # Team B

    # 3. Team A Goalkeeper in goal area (X=15m, dark kit)
    gk_dark_hsv = (80.0, 30.0, 50.0)
    assert classifier.predict_role(gk_dark_hsv, pos_m=(15.0, 34.0), is_deepest_a=True) == 3  # Team A GK

    # 4. Outfield Referee (X=45m, dark kit)
    ref_dark_hsv = (80.0, 30.0, 50.0)
    assert classifier.predict_role(ref_dark_hsv, pos_m=(45.0, 34.0)) == 2  # Referee

    # 5. Dugout Coach standing at bottom bench (y2 = 890px)
    assert classifier.predict_role(white_hsv, pos_m=(55.0, -5.0), box_y2=890.0) == 5  # Coach


def test_rolling_majority_vote_persistence():
    """
    Test that temporary noise is filtered out by majority voting.
    """
    classifier = TeamClassifier(history_window=10)
    tid = 10

    for _ in range(7):
        role = classifier.update_track_role(tid, 0)
    assert role == 0

    classifier.update_track_role(tid, 1)
    role_after_noise = classifier.update_track_role(tid, 1)
    assert role_after_noise == 0
