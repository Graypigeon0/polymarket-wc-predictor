"""
Monte Carlo tournament simulator.

Runs N (default 10,000) full WC 2026 simulations from current state:
  1. Sample each remaining group-stage match from its Dixon-Coles distribution
  2. Resolve group standings with FIFA tiebreaker chain
  3. Build knockout bracket per FIFA seeding rules
  4. Simulate knockouts (regulation -> extra time -> shootout)
  5. Aggregate: champion, finalist, top-4 by team; per-stage probabilities

WC 2026 format (new for this tournament):
  - 48 teams, 12 groups of 4
  - Top 2 from each group + 8 best 3rd-placed teams = 32 to Round of 32
  - Then R16 -> QF -> SF -> Final (single-elimination)

FIFA tiebreaker chain for group standings:
  1. Points
  2. Goal difference
  3. Goals scored
  4. Head-to-head points (among tied teams)
  5. Head-to-head goal difference
  6. Head-to-head goals scored
  7. Fair play (yellow/red card count) -- skipped, modelled as random
  8. Drawing of lots -- random
"""

from __future__ import annotations

import numpy as np
import structlog

from backend.config import get_settings
from backend.models.dixon_coles import score_grid

log = structlog.get_logger()


def sample_score(lh: float, la: float, rho: float, rng: np.random.Generator,
                 max_goals: int = 8) -> tuple[int, int]:
    """Sample a single match scoreline from the Dixon-Coles distribution."""
    grid = score_grid(lh, la, rho, max_goals)
    flat = grid.flatten()
    idx = rng.choice(len(flat), p=flat / flat.sum())
    return divmod(idx, max_goals + 1)


def sample_knockout(lh: float, la: float, rho: float, rng: np.random.Generator) -> int:
    """
    Sample a knockout winner: 0 = home, 1 = away.
    Regulation -> if draw, extra time (lambdas * 0.33) -> if still draw, 50/50 shootout
    (could be tilted slightly by GK rating later).
    """
    h, a = sample_score(lh, la, rho, rng)
    if h > a:
        return 0
    if a > h:
        return 1
    # extra time
    h2, a2 = sample_score(lh * 0.33, la * 0.33, rho, rng, max_goals=4)
    if h2 > a2:
        return 0
    if a2 > h2:
        return 1
    return int(rng.random() < 0.5)


async def simulate() -> None:
    """Main entry — run N simulations and persist aggregated probabilities."""
    s = get_settings()
    log.info("tournament_sim.start", n_runs=s.monte_carlo_runs)
    # TODO:
    #   1. Load current state from db:
    #        - 48 teams with effective ratings (base + squad + active deltas)
    #        - already-played group matches (use real results, don't resample)
    #        - remaining fixtures
    #   2. Run s.monte_carlo_runs simulations
    #   3. For each: resolve groups, build bracket, play knockouts
    #   4. Aggregate per-team {p_win_outright, p_reach_final, p_reach_semi, ...}
    #   5. Upsert tournament_predictions rows
    log.info("tournament_sim.todo")


# ---------------------------------------------------------------------
# FIFA tiebreaker helpers (pure functions, easy to unit test)
# ---------------------------------------------------------------------

def resolve_group(results: list[dict]) -> list[str]:
    """
    Given list of {home, away, hg, ag} results for a single group of 4,
    return team codes ordered by FIFA tiebreaker chain.
    """
    # TODO: implement full tiebreaker chain; unit test against historical groups
    raise NotImplementedError
