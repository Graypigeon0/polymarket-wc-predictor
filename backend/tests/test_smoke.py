"""Smoke tests — verify nothing's broken at import time."""

from __future__ import annotations

import numpy as np


def test_imports():
    """All modules import without error."""
    from backend import config, main  # noqa: F401
    from backend.alerts import telegram  # noqa: F401
    from backend.edges import calculator  # noqa: F401
    from backend.ingestion import fbref, football_data, polymarket, sofascore  # noqa: F401
    from backend.ingestion.news import reddit, rss  # noqa: F401
    from backend.models import dixon_coles, llm_context, squad_strength, top_scorer, tournament_sim  # noqa: F401


def test_dixon_coles_probabilities_sum_to_one():
    """Sanity check: match probabilities are a valid distribution."""
    from backend.models.dixon_coles import predict_match

    # Two equally-rated teams, neutral venue
    pred = predict_match(attack_h=0.0, defense_h=0.0,
                         attack_a=0.0, defense_a=0.0,
                         rho=-0.05)

    total = pred.p_home + pred.p_draw + pred.p_away
    assert abs(total - 1.0) < 1e-3, f"probs sum to {total}, expected 1.0"

    # Symmetry: equal teams should give equal home/away probs
    assert abs(pred.p_home - pred.p_away) < 1e-3


def test_dixon_coles_stronger_team_wins_more():
    """If team A has much higher attack, they should be favored."""
    from backend.models.dixon_coles import predict_match

    pred = predict_match(attack_h=0.5, defense_h=-0.3,
                         attack_a=-0.5, defense_a=0.3,
                         rho=-0.05)
    assert pred.p_home > pred.p_away


def test_score_grid_sums_to_one():
    from backend.models.dixon_coles import score_grid
    grid = score_grid(1.5, 1.2, -0.05)
    assert abs(grid.sum() - 1.0) < 1e-6
    assert np.all(grid >= -1e-9)   # no negative probs after renormalisation


def test_squad_strength_handles_empty():
    from backend.models.squad_strength import compute_team_rating
    a, d = compute_team_rating([])
    assert a == 0.0 and d == 0.0
