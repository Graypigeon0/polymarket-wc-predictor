"""
Backtest harness — validate model calibration on a past tournament.

Method (no lookahead):
  1. Choose a target tournament (e.g. EURO2024) and its start date.
  2. Fit Dixon-Coles using ONLY matches before that date.
  3. Predict 1X2 for every match in the target tournament.
  4. Score predictions against actual results.

Metrics:
  - Brier score (multiclass): mean squared error of probability vector vs
    one-hot outcome. Lower is better. Baseline (always predict base rates)
    ~0.62; a good model ~0.55-0.58.
  - Log loss: penalizes confident wrong predictions. Lower is better.
  - Accuracy: fraction where argmax(prob) == actual outcome.
  - Calibration table: bin predictions, compare predicted vs observed frequency.

If the model is well-calibrated, its disagreements with the market are
genuine edges. If not, they are noise.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

from backend.db.client import get_client
from backend.models import dixon_coles

log = structlog.get_logger()


# Known tournament start dates for backtesting
TOURNAMENT_STARTS = {
    "EURO2024": "2024-06-14",
    "WC2022":   "2022-11-20",
    "COPA2024": "2024-06-20",
}


def _result_index(home_goals: int, away_goals: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def _brier_multiclass(probs: list[float], outcome_idx: int) -> float:
    """Multiclass Brier score for a single prediction."""
    target = [0.0, 0.0, 0.0]
    target[outcome_idx] = 1.0
    return sum((probs[i] - target[i]) ** 2 for i in range(3))


def _log_loss(probs: list[float], outcome_idx: int) -> float:
    p = max(probs[outcome_idx], 1e-12)
    return -math.log(p)


async def run_backtest(target: str = "EURO2024") -> dict[str, Any]:
    """Fit as-of the target tournament, predict its matches, score them."""
    if target not in TOURNAMENT_STARTS:
        return {"error": f"unknown target {target}",
                "known": list(TOURNAMENT_STARTS)}

    as_of = TOURNAMENT_STARTS[target]
    as_of_dt = datetime.fromisoformat(as_of).replace(tzinfo=timezone.utc)
    log.info("backtest.start", target=target, as_of=as_of)

    # 1. Load training matches strictly before the tournament, excluding the
    #    target tournament itself and the live WC2026 fixtures.
    matches = dixon_coles._load_training_matches(
        as_of=as_of, exclude_competitions=("WC2026", target))
    if len(matches) < 50:
        return {"error": "not enough pre-tournament data", "n": len(matches)}

    log.info("backtest.training_data", n_matches=len(matches))

    # 2. Fit in-memory (no DB writes), shrinking toward FIFA-ranking priors
    prior_net = dixon_coles._load_team_priors()
    fifa_pts = dixon_coles._load_team_fifa_points()
    fitted = dixon_coles._fit_core(matches, as_of_dt, prior_net, fifa_pts)
    team_idx = fitted["team_idx"]
    attacks, defenses = fitted["attacks"], fitted["defenses"]
    rho = fitted["rho"]

    # 3. Load target tournament matches (actual results)
    db = get_client()
    target_matches = (db.table("matches")
                      .select("home_id,away_id,home_goals,away_goals,stage")
                      .eq("competition", target)
                      .eq("completed", True)
                      .execute()).data or []

    if not target_matches:
        return {"error": f"no completed matches for {target}"}

    # 4. Predict + score
    briers, loglosses = [], []
    correct = 0
    scored = 0
    skipped_no_rating = 0
    # Calibration bins (decile buckets on predicted prob of the realised class)
    calib_bins = {i: {"sum_pred": 0.0, "hits": 0, "n": 0} for i in range(10)}

    for m in target_matches:
        hi = team_idx.get(m["home_id"])
        ai = team_idx.get(m["away_id"])
        if hi is None or ai is None:
            skipped_no_rating += 1
            continue

        pred = dixon_coles.predict_match(
            attack_h=float(attacks[hi]), defense_h=float(defenses[hi]),
            attack_a=float(attacks[ai]), defense_a=float(defenses[ai]),
            rho=rho, home_adv=0.0,   # tournament matches are neutral
        )
        probs = [pred.p_home, pred.p_draw, pred.p_away]
        outcome = _result_index(m["home_goals"], m["away_goals"])

        briers.append(_brier_multiclass(probs, outcome))
        loglosses.append(_log_loss(probs, outcome))
        if int(np.argmax(probs)) == outcome:
            correct += 1
        scored += 1

        # Calibration: bucket by the predicted probability assigned to each
        # of the three outcomes, tracking whether that outcome occurred.
        for cls in range(3):
            b = min(int(probs[cls] * 10), 9)
            calib_bins[b]["sum_pred"] += probs[cls]
            calib_bins[b]["hits"] += (1 if outcome == cls else 0)
            calib_bins[b]["n"] += 1

    if scored == 0:
        return {"error": "no matches could be scored (no overlapping teams)",
                "skipped": skipped_no_rating}

    # Baseline Brier: always predict historical base rates (home/draw/away)
    # Rough international base rates: home 0.45, draw 0.27, away 0.28 — but
    # tournament matches are neutral, so use 0.38/0.27/0.35.
    base = [0.38, 0.27, 0.35]
    baseline_briers = []
    for m in target_matches:
        if team_idx.get(m["home_id"]) is None or team_idx.get(m["away_id"]) is None:
            continue
        outcome = _result_index(m["home_goals"], m["away_goals"])
        baseline_briers.append(_brier_multiclass(base, outcome))

    # Build calibration table
    calib_table = []
    for b in range(10):
        cb = calib_bins[b]
        if cb["n"] > 0:
            calib_table.append({
                "bin":            f"{b*10}-{b*10+10}%",
                "avg_predicted":  round(cb["sum_pred"] / cb["n"], 3),
                "observed_freq":  round(cb["hits"] / cb["n"], 3),
                "n":              cb["n"],
            })

    result = {
        "target":            target,
        "as_of":             as_of,
        "training_matches":  len(matches),
        "scored_matches":    scored,
        "skipped_no_rating": skipped_no_rating,
        "brier":             round(float(np.mean(briers)), 4),
        "baseline_brier":    round(float(np.mean(baseline_briers)), 4),
        "log_loss":          round(float(np.mean(loglosses)), 4),
        "accuracy":          round(correct / scored, 3),
        "calibration":       calib_table,
    }
    log.info("backtest.done", **{k: v for k, v in result.items()
                                 if k != "calibration"})
    return result
