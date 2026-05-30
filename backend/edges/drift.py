"""
Closing-line drift analysis.

For each market we've alerted on, compare the Polymarket price AT THE TIME
WE ALERTED vs. the LATEST price. The direction of movement tells us whether
our model is anticipating or lagging:

  - If market moved TOWARD our model_prob -> the market converged with us
    (good sign: we found a mispricing before the market did)
  - If market moved AWAY from our model_prob -> the market diverged
    (bad sign: we're disagreeing with collective wisdom in the wrong direction)

This is one of the strongest sanity checks on a betting model. Sharp models
consistently lead the closing line; weak models trail it.
"""

from __future__ import annotations

from typing import Any

import structlog

from backend.db.client import get_client

log = structlog.get_logger()


async def compute_drift_summary() -> dict[str, Any]:
    """
    For each market with at least one alert, compute:
      - model_prob at alert (what we said)
      - pm_prob at alert (what market said when we alerted)
      - latest pm_prob (where market is now)
      - drift = latest - alert (positive = market moved up; negative = down)
      - direction = 'toward'/'away' depending on whether the market moved
        toward our model's view

    Returns aggregate stats and per-market detail.
    """
    db = get_client()

    # First alert per market (the moment we'd have bet)
    edges = (db.table("edges")
             .select("market_id,model_prob,pm_prob,edge,alerted_at")
             .eq("alerted", True)
             .order("alerted_at")
             .execute()).data or []
    first_alerts: dict[str, dict] = {}
    for e in edges:
        if e["market_id"] not in first_alerts:
            first_alerts[e["market_id"]] = e

    if not first_alerts:
        return {"n_markets": 0, "per_market": []}

    market_ids = list(first_alerts.keys())

    # Latest price per market — use a per-market lookup since IN() + ORDER BY
    # gets one row per market with our pattern
    latest_prices: dict[str, float] = {}
    for mid in market_ids:
        r = (db.table("polymarket_prices")
             .select("price,captured_at")
             .eq("market_id", mid)
             .order("captured_at", desc=True)
             .limit(1)
             .execute()).data
        if r and r[0].get("price") is not None:
            latest_prices[mid] = float(r[0]["price"])

    # Market metadata for labels
    mkts = (db.table("polymarket_markets")
            .select("id,outcome_label,market_type")
            .in_("id", market_ids)
            .execute()).data or []
    label_by_id = {m["id"]: m for m in mkts}

    per_market = []
    n_toward = 0; n_away = 0; n_flat = 0
    total_drift_signed = 0.0  # positive = market moved toward us on average

    for mid, alert in first_alerts.items():
        label_row = label_by_id.get(mid, {})
        model_prob = float(alert["model_prob"])
        alert_pm = float(alert["pm_prob"])
        latest_pm = latest_prices.get(mid)
        if latest_pm is None:
            continue

        market_drift = latest_pm - alert_pm   # raw price movement
        # "Toward us" means the market moved in the direction of our model.
        # Our model said model_prob; if model_prob > alert_pm, we said HIGHER —
        # so a positive drift (market went up) = toward us. Vice-versa.
        sign_toward = 1.0 if model_prob > alert_pm else -1.0
        signed_drift = market_drift * sign_toward

        if abs(signed_drift) < 0.005:
            direction = "flat"
            n_flat += 1
        elif signed_drift > 0:
            direction = "toward"
            n_toward += 1
        else:
            direction = "away"
            n_away += 1
        total_drift_signed += signed_drift

        per_market.append({
            "market":    label_row.get("outcome_label", mid[:24]),
            "type":      label_row.get("market_type", "?"),
            "model":     round(model_prob, 3),
            "alert_pm":  round(alert_pm, 3),
            "latest_pm": round(latest_pm, 3),
            "drift_pp":  round(market_drift * 100, 1),
            "direction": direction,
        })

    n_total = n_toward + n_away + n_flat
    avg_signed_drift_pp = (total_drift_signed / n_total * 100) if n_total else 0.0
    pct_toward = (n_toward / n_total) if n_total else 0.0

    # Sort: biggest move toward us first, then biggest move away (most informative)
    per_market.sort(key=lambda r: r["drift_pp"], reverse=True)

    return {
        "n_markets":           n_total,
        "n_toward":            n_toward,
        "n_away":              n_away,
        "n_flat":              n_flat,
        "pct_toward":          round(pct_toward, 3),
        "avg_signed_drift_pp": round(avg_signed_drift_pp, 2),
        "per_market":          per_market,
    }
