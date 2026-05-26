"""
Squad strength adjuster.

Converts a 26-man squad into dynamic team attack/defense ratings by weighting
each player's underlying stats by their probability of starting.

Final team ratings = base Dixon-Coles ratings + squad adjustment + LLM context deltas.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger()

# Position weights for aggregating individual stats into team rating.
# Tuned on backtests; these are starting values.
POSITION_WEIGHTS_ATTACK = {"GK": 0.00, "DEF": 0.10, "MID": 0.30, "ATT": 0.60}
POSITION_WEIGHTS_DEFENSE = {"GK": 0.20, "DEF": 0.50, "MID": 0.25, "ATT": 0.05}


def compute_team_rating(squad: list[dict]) -> tuple[float, float]:
    """
    Args:
        squad: list of dicts with keys
          {position, starter_prob, fitness, club_xg90, club_xga90, intl_g90}

    Returns:
        (attack_rating_adjustment, defense_rating_adjustment)
        These are added to the team's Dixon-Coles base rating.
    """
    attack = 0.0
    defense = 0.0
    for p in squad:
        if p.get("status") not in ("available", "doubtful"):
            continue
        avail = p["starter_prob"] * p["fitness"]
        wA = POSITION_WEIGHTS_ATTACK.get(p["position"], 0.0)
        wD = POSITION_WEIGHTS_DEFENSE.get(p["position"], 0.0)

        # Blend club + international form
        attack_contrib = (
            0.7 * (p.get("club_xg90") or 0.0)
            + 0.3 * (p.get("intl_g90") or 0.0)
        )
        defense_contrib = -(p.get("club_xga90") or 0.0)   # lower xGA = better

        attack  += avail * wA * attack_contrib
        defense += avail * wD * defense_contrib

    return attack, defense


async def recompute_team_ratings() -> None:
    """
    For each WC team: pull current squad, compute attack/defense adjustments,
    combine with base Dixon-Coles ratings + active LLM context deltas,
    cache the resulting effective ratings for use by match predictor.
    """
    log.info("squad_strength.recompute.todo")
    # TODO: pull squads from db, run compute_team_rating, write to cache table
