"""
Dixon-Coles bivariate Poisson match model.

Reference:
  Dixon, M.J. & Coles, S.G. (1997). Modelling Association Football Scores
  and Inefficiencies in the Football Betting Market.

For each team we fit:
  - attack[i]:  log-rate of scoring
  - defense[i]: log-rate of conceding (lower = better defense)
  - home_adv:   global home advantage
  - rho:        low-score correlation parameter (Dixon-Coles tau function)

Expected goals (neutral venue):
  lambda_h = exp(attack[h] + defense[a])
  lambda_a = exp(attack[a] + defense[h])

Joint score: tau(x,y,lambda_h,lambda_a,rho) * Poisson(x;lambda_h) * Poisson(y;lambda_a)

Fitted with weighted maximum likelihood:
  - Exponential time decay (half-life 18 months)
  - Competition weight (WC=1.0, EURO/COPA=0.85, qualifier=0.7, friendly=0.4)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog
from scipy.optimize import minimize
from scipy.stats import poisson

from backend.config import get_settings
from backend.db.client import get_client

log = structlog.get_logger()


# ---------------------------------------------------------------------
# Public prediction container
# ---------------------------------------------------------------------

@dataclass
class MatchPrediction:
    p_home: float
    p_draw: float
    p_away: float
    expected_home_goals: float
    expected_away_goals: float
    score_distribution: dict[str, float]


# ---------------------------------------------------------------------
# Dixon-Coles tau function (low-score correlation correction)
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
    s = grid.sum()
    if s > 0:
        grid /= s
    return grid


def predict_match(
    attack_h: float,
    defense_h: float,
    attack_a: float,
    defense_a: float,
    rho: float = -0.05,
    home_adv: float = 0.0,
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
        if grid[x, y] > 0.005
    }

    return MatchPrediction(
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        expected_home_goals=lh, expected_away_goals=la,
        score_distribution=score_dist,
    )


# ---------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------

# Competition weights for weighted MLE
# Competition weights for weighted MLE. Major tournaments weight = 1.0;
# continental cups slightly lower; qualifiers and Nations League fairly high
# (competitive, current squads); friendlies low (rotated lineups, low stakes).
COMPETITION_WEIGHTS: dict[str, float] = {
    # Tournament finals (handled by openfootball)
    "WC2026":     1.00,
    "WC2022":     1.00,
    "WC2018":     1.00,
    "WC2014":     1.00,
    "EURO2024":   0.85,
    "EURO2020":   0.85,
    # From martj42 dataset
    "WCMAIN":     1.00,    # WC matches not in openfootball
    "WCQUAL":     0.75,    # World Cup qualifiers
    "EURMAIN":    0.85,
    "EURQUAL":    0.70,
    "UNL":        0.70,    # UEFA Nations League
    "CNL":        0.65,    # CONCACAF Nations League
    "COPA":       0.85,
    "AFCON":      0.70,
    "AFCONQUAL":  0.55,
    "ASIAN":      0.65,
    "ASIANQUAL":  0.50,
    "GOLDCUP":    0.70,
    "OFC":        0.50,
    "FRIENDLY":   0.40,
    "OTHER":      0.30,
}

# Half-life for exponential time decay (months)
RECENCY_HALF_LIFE_MONTHS = 18.0


def _recency_weight(match_date: datetime, ref_date: datetime) -> float:
    """Exponential decay weight: 1.0 at ref_date, halves every 18 months."""
    months_old = (ref_date - match_date).days / 30.44
    return 0.5 ** (max(0.0, months_old) / RECENCY_HALF_LIFE_MONTHS)


def _load_training_matches() -> list[dict[str, Any]]:
    """Pull completed historical matches from Supabase."""
    db = get_client()
    # Only completed matches with scores; skip WC2026 (current tournament, no results yet)
    r = (db.table("matches")
         .select("home_id,away_id,kickoff,home_goals,away_goals,competition")
         .eq("completed", True)
         .neq("competition", "WC2026")
         .execute())
    return r.data or []


def _build_team_index(matches: list[dict]) -> dict[str, int]:
    """Assign each team UUID to a contiguous integer index 0..N-1."""
    teams = sorted({m["home_id"] for m in matches} | {m["away_id"] for m in matches})
    return {tid: i for i, tid in enumerate(teams)}


# L2 regularization strength (Bayesian-style shrinkage toward global mean = 0).
# Higher = stronger pull toward equality. ~8 is moderate for ~60 teams / ~300 matches.
REG_LAMBDA = 8.0


def _neg_log_likelihood(
    params: np.ndarray,
    matches: list[dict],
    team_idx: dict[str, int],
    n_teams: int,
    ref_date: datetime,
) -> float:
    """Negative log-likelihood + L2 regularization."""
    attacks = params[:n_teams]
    defenses = params[n_teams:2 * n_teams]
    home_adv = params[-2]
    rho = params[-1]

    total = 0.0
    for m in matches:
        i = team_idx[m["home_id"]]
        j = team_idx[m["away_id"]]
        x = int(m["home_goals"])
        y = int(m["away_goals"])

        lh = math.exp(attacks[i] + defenses[j] + home_adv)
        la = math.exp(attacks[j] + defenses[i])
        lh = min(lh, 8.0)
        la = min(la, 8.0)

        tau = _tau(x, y, lh, la, rho)
        log_p = (
            x * math.log(lh) - lh - math.lgamma(x + 1)
            + y * math.log(la) - la - math.lgamma(y + 1)
            + math.log(max(tau, 1e-10))
        )

        kickoff = datetime.fromisoformat(m["kickoff"].replace("Z", "+00:00"))
        w_recency = _recency_weight(kickoff, ref_date)
        w_comp = COMPETITION_WEIGHTS.get(m["competition"], 0.5)
        w = w_recency * w_comp

        total -= w * log_p

    # L2 regularization: pull attack/defense toward 0 (the global mean,
    # since sum-to-zero constraint is enforced). This stabilises ratings
    # for teams with few matches in the training set.
    reg = REG_LAMBDA * float(np.sum(attacks ** 2) + np.sum(defenses ** 2))
    return total + reg


async def fit() -> dict[str, Any]:
    """
    Fit Dixon-Coles attack/defense ratings via weighted MLE.

    Persists fitted ratings to `teams.base_attack`, `teams.base_defense`.
    Returns fit summary: {teams, n_matches, rho, home_adv, nll}.
    """
    log.info("dixon_coles.fit.start")
    matches = _load_training_matches()
    if len(matches) < 50:
        log.error("dixon_coles.fit.too_few_matches", n=len(matches))
        return {"error": "not enough matches", "n": len(matches)}

    team_idx = _build_team_index(matches)
    n_teams = len(team_idx)
    log.info("dixon_coles.fit.data", n_matches=len(matches), n_teams=n_teams)

    # Initial guess: all attacks/defenses 0, home_adv = 0.1, rho = -0.05
    x0 = np.zeros(2 * n_teams + 2)
    x0[-2] = 0.1     # home_adv
    x0[-1] = -0.05   # rho

    ref_date = datetime.now(timezone.utc)

    # Sum-to-zero constraint on attack ratings (identifiability)
    constraint = {"type": "eq", "fun": lambda p: float(np.sum(p[:n_teams]))}
    # Reasonable bounds to keep optimisation stable
    bounds = [(-3.0, 3.0)] * (2 * n_teams) + [(-0.5, 1.0), (-0.2, 0.2)]

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(matches, team_idx, n_teams, ref_date),
        method="SLSQP",
        bounds=bounds,
        constraints=[constraint],
        options={"maxiter": 200, "ftol": 1e-5},
    )

    if not result.success:
        log.warning("dixon_coles.fit.optimizer_warning", msg=result.message)

    attacks = result.x[:n_teams]
    defenses = result.x[n_teams:2 * n_teams]
    home_adv = float(result.x[-2])
    rho = float(result.x[-1])

    # Persist back to Supabase
    db = get_client()
    updated = 0
    for team_id, idx in team_idx.items():
        db.table("teams").update({
            "base_attack":  float(attacks[idx]),
            "base_defense": float(defenses[idx]),
            "home_adv":     home_adv,
        }).eq("id", team_id).execute()
        updated += 1

    log.info("dixon_coles.fit.done",
             n_teams=n_teams, n_matches=len(matches),
             home_adv=home_adv, rho=rho,
             updated=updated, nll=float(result.fun))

    return {
        "n_teams":    n_teams,
        "n_matches":  len(matches),
        "home_adv":   home_adv,
        "rho":        rho,
        "nll":        float(result.fun),
        "converged":  bool(result.success),
        "rho_global": rho,  # store as engine-wide constant if needed later
    }


async def predict_upcoming() -> int:
    """
    For every uncompleted match in `matches`, compute and upsert a
    match_predictions row using the latest fitted team ratings.
    Returns count of predictions written.
    """
    s = get_settings()
    db = get_client()

    # Load fitted ratings
    teams_resp = db.table("teams").select("id,base_attack,base_defense,home_adv").execute()
    ratings = {t["id"]: t for t in (teams_resp.data or [])
               if t.get("base_attack") is not None}

    if not ratings:
        log.warning("dixon_coles.predict.no_ratings_fit_first")
        return 0

    # Use a shared rho — pulled from any one team's home_adv... actually we store rho per-engine.
    # For simplicity use a constant fitted value; we re-fit periodically.
    rho = -0.05
    home_adv_global = next(iter(ratings.values())).get("home_adv", 0.0) or 0.0

    upcoming = (db.table("matches")
                .select("id,home_id,away_id,competition,is_neutral")
                .eq("completed", False)
                .execute())

    written = 0
    for m in upcoming.data or []:
        h = ratings.get(m["home_id"])
        a = ratings.get(m["away_id"])
        if not h or not a:
            continue  # missing fit for this team (e.g. new country)

        home_adv = 0.0 if m.get("is_neutral", True) else home_adv_global
        pred = predict_match(
            attack_h=h["base_attack"], defense_h=h["base_defense"],
            attack_a=a["base_attack"], defense_a=a["base_defense"],
            rho=rho, home_adv=home_adv,
        )

        # Delete any prior prediction for this match (keeps the table
        # idempotent — re-running predict overwrites instead of duplicating).
        db.table("match_predictions").delete().eq("match_id", m["id"]).execute()
        db.table("match_predictions").insert({
            "match_id":            m["id"],
            "model_version":       s.model_version,
            "p_home":              pred.p_home,
            "p_draw":              pred.p_draw,
            "p_away":              pred.p_away,
            "expected_home_goals": pred.expected_home_goals,
            "expected_away_goals": pred.expected_away_goals,
            "score_distribution":  pred.score_distribution,
        }).execute()
        written += 1

    log.info("dixon_coles.predict.done", written=written)
    return written
