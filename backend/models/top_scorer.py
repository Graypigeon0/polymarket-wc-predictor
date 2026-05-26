"""
Top scorer / Golden Boot model.

Player-level Poisson on goals per 90 minutes, fitted on club + international goals
(recency-weighted, competition-strength adjusted).

For each Monte Carlo run from tournament_sim:
  - For each match the player's team plays, sample their minutes (starter prob)
  - Sample goals from Poisson(rate * minutes/90)
  - Accumulate tournament total

Top scorer probability = fraction of MC runs where player ends with max goals
(with tied-finish handling).
"""

from __future__ import annotations

import structlog

log = structlog.get_logger()


def player_goal_rate(
    club_g90: float,
    intl_g90: float,
    penalty_taker: bool = False,
    set_piece_taker: bool = False,
) -> float:
    """
    Blend club + international scoring rates; bump for penalty/set-piece duty.
    """
    base = 0.65 * club_g90 + 0.35 * intl_g90
    if penalty_taker:
        base += 0.10
    if set_piece_taker:
        base += 0.03
    return max(0.0, base)


async def predict() -> None:
    """
    Run player-level sim on top of tournament_sim's bracket samples.

    Plan:
      1. Pull squad attackers + key midfielders with non-zero goal rate
      2. Re-run tournament sim (or hook into stored bracket samples)
      3. For each run, for each player, sample minutes -> sample goals -> total
      4. p_top_scorer[player] = fraction of runs where their total is max
         (split ties evenly across tied players)
      5. Upsert top_scorer_predictions rows
    """
    log.info("top_scorer.predict.todo")
    # TODO: implement
