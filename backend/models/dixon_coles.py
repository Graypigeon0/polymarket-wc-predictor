"""
Dixon-Coles bivariate Poisson match model with robust iteratively
reweighted MLE for opponent-strength bias correction.

Reference:
  Dixon, M.J. & Coles, S.G. (1997). Modelling Association Football Scores
  and Inefficiencies in the Football Betting Market.

Per team we fit attack[i] (log scoring rate) and defense[i] (log conceding
rate), plus global home_adv and Dixon-Coles rho.

Robust fit overview:
  1. Initial MLE with recency + competition weights
  2. Compute Pearson-style residuals for every match
  3. Tukey bisquare weights downweight anomalous matches
     (e.g. Spain 7-0 Andorra, where actual goals far exceeded prediction)
  4. Refit with the new weights * original weights
  5. Iterate until ratings stabilise (~5 iterations)

This corrects opponent-strength bias: easy blowouts against weak qualifiers
no longer dominate the elite teams' attack ratings.
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
# Dixon-Coles tau function (low-score correlation)
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
# Configuration
# ---------------------------------------------------------------------

COMPETITION_WEIGHTS: dict[str, float] = {
    "WC2026":     1.00,
    "WC2022":     1.00,
    "WC2018":     1.00,
    "WC2014":     1.00,
    "EURO2024":   0.85,
    "EURO2020":   0.85,
    "WCMAIN":     1.00,
    "WCQUAL":     0.75,
    "EURMAIN":    0.85,
    "EURQUAL":    0.70,
    "UNL":        0.70,
    "CNL":        0.65,
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

RECENCY_HALF_LIFE_MONTHS = 18.0

# L2 regularization strength (shrinks ratings toward 0)
REG_LAMBDA = 8.0

# Robust regression parameters
ROBUST_ITERATIONS = 5            # number of IRLS passes
TUKEY_C = 4.0                    # bisquare cutoff; smaller = more aggressive downweighting
MIN_RESIDUAL_WEIGHT = 0.05       # floor so no match is fully excluded


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def _recency_weight(match_date: datetime, ref_date: datetime) -> float:
    months_old = (ref_date - match_date).days / 30.44
    return 0.5 ** (max(0.0, months_old) / RECENCY_HALF_LIFE_MONTHS)


def _load_training_matches() -> list[dict[str, Any]]:
    db = get_client()
    # Paginate — Supabase default limit is 1000 rows per query.
    all_rows: list[dict[str, Any]] = []
    page_size = 1000
    offset = 0
    while True:
        r = (db.table("matches")
             .select("home_id,away_id,kickoff,home_goals,away_goals,competition")
             .eq("completed", True)
             .neq("competition", "WC2026")
             .range(offset, offset + page_size - 1)
             .execute())
        rows = r.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


def _build_team_index(matches: list[dict]) -> dict[str, int]:
    teams = sorted({m["home_id"] for m in matches} | {m["away_id"] for m in matches})
    return {tid: i for i, tid in enumerate(teams)}


# ---------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------

def _neg_log_likelihood(
    params: np.ndarray,
    matches: list[dict],
    team_idx: dict[str, int],
    n_teams: int,
    match_weights: np.ndarray,
) -> float:
    attacks = params[:n_teams]
    defenses = params[n_teams:2 * n_teams]
    home_adv = params[-2]
    rho = params[-1]

    total = 0.0
    for k, m in enumerate(matches):
        w = float(match_weights[k])
        if w < 1e-9:
            continue
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

        total -= w * log_p

    # L2 regularization (shrinkage toward 0)
    reg = REG_LAMBDA * float(np.sum(attacks ** 2) + np.sum(defenses ** 2))
    return total + reg


# ---------------------------------------------------------------------
# Residual-based robust weights
# ---------------------------------------------------------------------

def _pearson_residuals(
    params: np.ndarray,
    matches: list[dict],
    team_idx: dict[str, int],
    n_teams: int,
) -> np.ndarray:
    """Per-match Pearson residual on the goal-difference axis."""
    attacks = params[:n_teams]
    defenses = params[n_teams:2 * n_teams]
    home_adv = params[-2]

    residuals = np.zeros(len(matches))
    for k, m in enumerate(matches):
        i = team_idx[m["home_id"]]
        j = team_idx[m["away_id"]]
        x = int(m["home_goals"])
        y = int(m["away_goals"])

        lh = math.exp(attacks[i] + defenses[j] + home_adv)
        la = math.exp(attacks[j] + defenses[i])
        lh = min(lh, 8.0)
        la = min(la, 8.0)

        # Standardised Pearson residual for total goals
        actual_total = x + y
        expected_total = lh + la
        sd = math.sqrt(max(expected_total, 0.5))
        residuals[k] = (actual_total - expected_total) / sd
    return residuals


def _tukey_weights(residuals: np.ndarray, c: float = TUKEY_C) -> np.ndarray:
    """Tukey bisquare: smooth zero outside [-c, c]. Used to downweight outliers."""
    r = np.abs(residuals) / c
    w = np.where(r < 1.0, (1.0 - r ** 2) ** 2, 0.0)
    return np.maximum(w, MIN_RESIDUAL_WEIGHT)


# ---------------------------------------------------------------------
# Fitting (robust IRLS)
# ---------------------------------------------------------------------

async def fit() -> dict[str, Any]:
    """Fit Dixon-Coles ratings with robust iteratively reweighted MLE."""
    log.info("dixon_coles.fit.start")
    matches = _load_training_matches()
    if len(matches) < 50:
        log.error("dixon_coles.fit.too_few_matches", n=len(matches))
        return {"error": "not enough matches", "n": len(matches)}

    team_idx = _build_team_index(matches)
    n_teams = len(team_idx)
    n_matches = len(matches)
    log.info("dixon_coles.fit.data", n_matches=n_matches, n_teams=n_teams)

    # Pre-compute static weights (recency + competition)
    ref_date = datetime.now(timezone.utc)
    base_weights = np.zeros(n_matches)
    for k, m in enumerate(matches):
        kickoff = datetime.fromisoformat(m["kickoff"].replace("Z", "+00:00"))
        wr = _recency_weight(kickoff, ref_date)
        wc = COMPETITION_WEIGHTS.get(m["competition"], 0.4)
        base_weights[k] = wr * wc

    # Initial params: zeros + home_adv 0.1, rho -0.05
    x0 = np.zeros(2 * n_teams + 2)
    x0[-2] = 0.1
    x0[-1] = -0.05

    constraint = {"type": "eq", "fun": lambda p: float(np.sum(p[:n_teams]))}
    bounds = [(-3.0, 3.0)] * (2 * n_teams) + [(-0.5, 1.0), (-0.2, 0.2)]

    params = x0
    nll_history = []
    weights = base_weights.copy()

    for iteration in range(ROBUST_ITERATIONS):
        # Looser tolerance for early iterations (we'll refine later)
        ftol = 1e-4 if iteration < ROBUST_ITERATIONS - 1 else 1e-6

        result = minimize(
            _neg_log_likelihood,
            params,
            args=(matches, team_idx, n_teams, weights),
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"maxiter": 200, "ftol": ftol},
        )
        params = result.x
        nll_history.append(float(result.fun))

        # Compute robust weights based on current fit
        resid = _pearson_residuals(params, matches, team_idx, n_teams)
        robust_w = _tukey_weights(resid)
        new_weights = base_weights * robust_w

        # Check convergence: how much did weights change?
        weight_change = float(np.abs(new_weights - weights).sum() / weights.sum())
        log.info("dixon_coles.fit.iteration",
                 iter=iteration + 1,
                 nll=nll_history[-1],
                 downweighted=int((robust_w < 0.5).sum()),
                 weight_change=weight_change)

        weights = new_weights
        if weight_change < 0.01:
            log.info("dixon_coles.fit.converged", iter=iteration + 1)
            break

    attacks = params[:n_teams]
    defenses = params[n_teams:2 * n_teams]
    home_adv = float(params[-2])
    rho = float(params[-1])

    # Persist ratings back to Supabase
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
             n_teams=n_teams, n_matches=n_matches,
             home_adv=home_adv, rho=rho,
             updated=updated, nll=nll_history[-1],
             iterations=len(nll_history))

    return {
        "n_teams":      n_teams,
        "n_matches":    n_matches,
        "home_adv":     home_adv,
        "rho":          rho,
        "nll":          nll_history[-1],
        "iterations":   len(nll_history),
        "nll_history":  nll_history,
    }


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

async def predict_upcoming() -> int:
    """Generate match_predictions rows for every uncompleted match."""
    s = get_settings()
    db = get_client()

    teams_resp = db.table("teams").select("id,base_attack,base_defense,home_adv").execute()
    ratings = {t["id"]: t for t in (teams_resp.data or [])
               if t.get("base_attack") is not None}

    if not ratings:
        log.warning("dixon_coles.predict.no_ratings_fit_first")
        return 0

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
            continue

        home_adv = 0.0 if m.get("is_neutral", True) else home_adv_global
        pred = predict_match(
            attack_h=h["base_attack"], defense_h=h["base_defense"],
            attack_a=a["base_attack"], defense_a=a["base_defense"],
            rho=rho, home_adv=home_adv,
        )

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
