"""
Edge outcome tracking — automated P&L on alerted bets.

Workflow:
  1. Daily, poll Polymarket via the same Gamma /events endpoint we use for
     discovery. Any market now `closed: true` with outcomePrices [1,0] or [0,1]
     is treated as resolved.
  2. Update polymarket_markets.resolved + resolution_outcome.
  3. Compute realised profit per $1 flat stake for each market we alerted on
     (using the first alert per market — we'd only bet once).

Profit math (Yes-side bet at price P):
  - Yes wins:  payoff per $1 staked = (1 - P) / P    (e.g. P=0.30 -> +$2.33)
  - Yes loses: payoff per $1 staked = -$1

Aggregate over all resolved markets gives total realised P&L, hit rate, ROI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from backend.db.client import get_client
from backend.ingestion.polymarket import _find_wc_events, _parse_json_field

log = structlog.get_logger()


async def refresh_resolutions() -> dict[str, int]:
    """Poll Gamma for our WC markets; mark any newly-resolved ones."""
    log.info("outcomes.refresh.start")
    db = get_client()

    # We only care about markets we've actually alerted on (saves work)
    alerted = (db.table("edges").select("market_id")
               .eq("alerted", True).execute()).data or []
    alerted_ids = {r["market_id"] for r in alerted}
    if not alerted_ids:
        log.info("outcomes.refresh.no_alerts_yet")
        return {"updated": 0, "alerts_tracked": 0}

    # Also fetch their current resolved flag (skip ones already resolved)
    existing = (db.table("polymarket_markets")
                .select("id,resolved")
                .in_("id", list(alerted_ids))
                .execute()).data or []
    already_resolved = {m["id"] for m in existing if m.get("resolved")}

    updated = 0
    async with httpx.AsyncClient() as client:
        events = await _find_wc_events(client)
        log.info("outcomes.refresh.events", n=len(events))

        for ev in events:
            for m in ev.get("markets", []):
                token_ids = _parse_json_field(m.get("clobTokenIds"))
                if not token_ids:
                    continue
                yes_token = str(token_ids[0])

                # Only check markets we care about + haven't already resolved
                if yes_token not in alerted_ids or yes_token in already_resolved:
                    continue

                if not bool(m.get("closed")):
                    continue

                prices = _parse_json_field(m.get("outcomePrices"))
                if not prices:
                    continue
                try:
                    yes_final = float(prices[0])
                except (TypeError, ValueError):
                    continue

                # Resolved markets settle at exactly 1.0 or 0.0
                if yes_final >= 0.99:
                    resolution = "Yes"
                elif yes_final <= 0.01:
                    resolution = "No"
                else:
                    continue  # closed but not yet settled (shouldn't happen often)

                db.table("polymarket_markets").update({
                    "resolved":           True,
                    "resolution_outcome": resolution,
                    "resolved_at":        datetime.now(timezone.utc).isoformat(),
                }).eq("id", yes_token).execute()
                updated += 1
                log.info("outcomes.resolved",
                         market=m.get("question", "")[:60], resolution=resolution)

    log.info("outcomes.refresh.done",
             updated=updated, alerts_tracked=len(alerted_ids))
    return {"updated": updated, "alerts_tracked": len(alerted_ids)}


async def compute_pnl() -> dict[str, Any]:
    """
    Compute realised P&L assuming $1 flat stake on the FIRST alert per market.
    Returns aggregate stats plus a per-market breakdown of resolved bets.
    """
    db = get_client()

    # Pull every alerted edge ordered by alert time (so we can dedupe to "first")
    edges = (db.table("edges")
             .select("market_id,model_prob,pm_prob,edge,alerted_at")
             .eq("alerted", True)
             .order("alerted_at")
             .execute()).data or []

    # First alert per market = the bet we'd have placed
    first_alerts: dict[str, dict] = {}
    for e in edges:
        if e["market_id"] not in first_alerts:
            first_alerts[e["market_id"]] = e

    if not first_alerts:
        return {"n_bets": 0, "n_resolved": 0, "n_pending": 0,
                "hit_rate": None, "total_profit": 0.0, "roi": None}

    # Look up market metadata + resolution
    market_ids = list(first_alerts.keys())
    markets = (db.table("polymarket_markets")
               .select("id,outcome_label,resolved,resolution_outcome,market_type")
               .in_("id", market_ids)
               .execute()).data or []
    m_by_id = {m["id"]: m for m in markets}

    n_resolved = 0; n_pending = 0; n_won = 0
    total_staked = 0.0; total_profit = 0.0
    per_market = []

    for mid, alert in first_alerts.items():
        m = m_by_id.get(mid, {})
        label = m.get("outcome_label") or mid[:24]
        pm_prob = float(alert["pm_prob"])

        if not m.get("resolved"):
            n_pending += 1
            per_market.append({"market": label, "status": "pending",
                               "model": round(alert["model_prob"], 3),
                               "pm": round(pm_prob, 3),
                               "edge": round(alert["edge"], 3)})
            continue

        n_resolved += 1
        total_staked += 1.0
        if m["resolution_outcome"] == "Yes" and pm_prob > 0:
            profit = (1.0 - pm_prob) / pm_prob
            total_profit += profit
            n_won += 1
            per_market.append({"market": label, "status": "won",
                               "pm": round(pm_prob, 3),
                               "profit": round(profit, 2)})
        else:
            total_profit -= 1.0
            per_market.append({"market": label, "status": "lost",
                               "pm": round(pm_prob, 3),
                               "profit": -1.0})

    return {
        "n_bets":         len(first_alerts),
        "n_resolved":     n_resolved,
        "n_pending":      n_pending,
        "n_won":          n_won,
        "hit_rate":       round(n_won / n_resolved, 3) if n_resolved else None,
        "total_staked":   round(total_staked, 2),
        "total_profit":   round(total_profit, 2),
        "roi":            round(total_profit / total_staked, 3) if total_staked else None,
        "per_market":     per_market,
    }
