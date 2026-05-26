"""
Dixon-Coles bivariate Poisson match model.

Reference:
  Dixon, M.J. & Coles, S.G. (1997). Modelling Association Football Scores
  and Inefficiencies in the Football Betting Market.

For each team we fit:
  - attack[i]:  log-rate of scoring
  - defense[i]: log-rate of conceding (lower = better defense)
  - home_adv:   global home advantage (~0.25 in club football, less for WC neutrals)
  - rho:        low-score correlation parameter (tau function)

Expected goals for match (home, away) at neutral venue:
  lambda_h = exp(attack[h] + defense[a])
  lambda_a = exp(attack[a] + defense[h])

Joint score P(X=x, Y=y) = tau(x, y, lambda_h, lambda_a, rho) * Poisson(x; lambda_h) * Poisson(y; lambda_a)

We fit with weighted MLE using:
  - exponential time decay (half-life 18 months)
  - competition weight (WC=1.0, EURO/COPA=0.85, qualifier=0.7, friendly=0.4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from scipy.stats import poisson

log = structlog.get_logger()


@dataclass
class MatchPrediction:
    p_home: float
    p_draw: float
    p_away: float
    expected_home_goals: float
    expected_away_goals: float
    score_distribution: dict[str, float]   # "h-a" -> P


# ---------------------------------------------------------------------
# Dixon-Coles tau function for low-score correlation
# ---------------------------------------------------------------------

def _tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - lh * la * rho
    if x == 0 and y == 1:
        return 1.0 + lh * rho
    if x == 1 and y == 0:
        return 1.0 + la * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_grid(lh: float, la: float, rho: float, max_goals: int = 8) -> np.ndarray:
    """Return a (max_goals+1) x (max_goals+1) matrix of joint score probs."""
    grid = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            grid[x, y] = (
                _tau(x, y, lh, la, rho)
                * poisson.pmf(x, lh)
                * poisson.pmf(y, la)
            )
    grid /= grid.sum()  # renormalise (tau-adjusted prob doesn't sum to 1)
    return grid


def predict_match(
    attack_h: float,
    defense_h: float,
    attack_a: float,
    defense_a: float,
    rho: float = -0.05,
    home_adv: float = 0.0,            # 0 for neutral WC venues
) -> MatchPrediction:
    lh = float(np.exp(attack_h + defense_a + home_adv))
    la = float(np.exp(attack_a + defense_h))
    grid = score_grid(lh, la, rho)

    p_home = float(np.tril(grid, -1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, 1).sum())

    score_dist = {
        f"{x}-{y}": float(grid[x, y])
        for x in range(grid.shape[0])
        for y in range(grid.shape[1])
        if grid[x, y] > 0.005   # only meaningful scores
    }

    return MatchPrediction(
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        expected_home_goals=lh, expected_away_goals=la,
        score_distribution=score_dist,
    )


# ---------------------------------------------------------------------
# Fitting (to be implemented)
# ---------------------------------------------------------------------

async def fit(history_matches: list[dict]) -> dict:
    """
    Fit attack/defense ratings via weighted MLE on historical matches.

    Returns dict of team_id -> {"attack": float, "defense": float}
    plus global {"rho": float, "home_adv": float}.
    """
    log.info("dixon_coles.fit.todo")
    # TODO:
    #   1. Build sparse parameter vector [attacks..., defenses..., rho, home_adv]
    #   2. Define neg-log-likelihood with recency + competition weights
    #   3. scipy.optimize.minimize (L-BFGS-B with constraint sum(attack)=0)
    #   4. Bootstrap n=200 for confidence bands
    raise NotImplementedError


async def predict_upcoming() -> None:
    """Run predict_match for every upcoming WC fixture, upsert match_predictions."""
    log.info("dixon_coles.predict_upcoming.todo")
    # TODO:
    #   1. Read upcoming matches from db
    #   2. Pull current team ratings (base + active deltas) from squad_strength
    #   3. Call predict_match
    #   4. Upsert match_predictions row
