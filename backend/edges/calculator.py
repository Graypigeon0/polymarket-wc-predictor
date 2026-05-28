"""
Edge calculator.

For each tracked Polymarket market with model coverage:
  - Pull latest pm_price → pm_prob
  - Look up model_prob (currently: match_1x2 markets only)
  - edge = model_prob - pm_prob
  - If positive AND outside cooldown window: insert into `edges` + fire Telegram

Markets we currently support:
  - match_1x2: linked via team_id; uses match_predictions table

Outright / group_winner / top_scorer markets require tournament simulation
or top-scorer model — not yet wired up. Those will be added once those
models exist.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog

from backend.alerts import telegram
from backend.config import get_settings
from backend.db.client import get_client

log = structlog.get_logger()


async def _latest_match_prediction(match_id: str, db) -> dict | None:
    r = (db.table("match_predictions")
         .select("p_home,p_draw,p_away,model_version,computed_at")
         .eq("match_id", match_id)
         .order("computed_at", desc=True)
         .limit(1)
         .execute())
    return (r.data or [None])[0]


async def _latest_price(token_id: str, db) -> dict | None:
    r = (db.table("polymarket_prices")
         .select("price,bid,ask,book_depth,captured_at")
         .eq("market_id", token_id)
         .order("captured_at", desc=True)
         .limit(1)
         .execute())
    return (r.data or [None])[0]


async def _was_recently_alerted(token_id: str, db, cooldown_min: int) -> bool:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=cooldown_min)).isoformat()
    r = (db.table("edges")
         .select("id")
         .eq("market_id", token_id)
         .eq("alerted", True)
         .gte("alerted_at", cutoff)
         .limit(1)
         .execute())
    return bool(r.data)


async def recompute_all() -> dict[str, int]:
    """
    Walk every active polymarket_market with model coverage, compute edge,
    write to edges table, fire alerts where appropriate.
    """
    s = get_settings()
    db = get_client()

    log.info("edges.recompute.start", threshold=s.edge_alert_threshold)
    markets = (db.table("polymarket_markets")
               .select("id,market_type,description,team_id,outcome_label")
               .eq("active", True)
               .in_("market_type", ["match_1x2"])
               .execute())
    rows = markets.data or []

    stats = {"considered": len(rows), "edged": 0, "alerted": 0, "no_model": 0, "no_price": 0}

    for m in rows:
        token_id = m["id"]

        # 1. Find latest price
        price_row = await _latest_price(token_id, db)
        if not price_row or price_row["price"] is None:
            stats["no_price"] += 1
            continue
        pm_prob = float(price_row["price"])

        # 2. Look up matching model prediction
        team_id = m.get("team_id")
        if not team_id:
            stats["no_model"] += 1
            continue

        # Find an upcoming match featuring this team. Pick the next one
        # by kickoff (the market description includes both teams; we use
        # team_id + nearest fixture as a heuristic mapping).
        match_q = (db.table("matches")
                   .select("id,home_id,away_id,kickoff,competition")
                   .eq("completed", False)
                   .or_(f"home_id.eq.{team_id},away_id.eq.{team_id}")
                   .eq("competition", "WC2026")
                   .order("kickoff")
                   .limit(1)
                   .execute())
        match_rows = match_q.data or []
        if not match_rows:
            stats["no_model"] += 1
            continue
        match = match_rows[0]

        pred = await _latest_match_prediction(match["id"], db)
        if not pred:
            stats["no_model"] += 1
            continue

        # Determine if this team is home or away → pick corresponding model prob
        if match["home_id"] == team_id:
            model_prob = float(pred["p_home"])
        else:
            model_prob = float(pred["p_away"])

        # 3. Compute edge
        edge = model_prob - pm_prob

        # 4. Always log it
        db.table("edges").insert({
            "market_id":     token_id,
            "model_prob":    model_prob,
            "pm_prob":       pm_prob,
            "edge":          edge,
            "edge_lower_ci": edge - 0.05,  # placeholder ±5pp band
            "model_version": pred["model_version"],
            "alerted":       False,
        }).execute()

        if edge > s.edge_alert_threshold:
            stats["edged"] += 1
            if await _was_recently_alerted(token_id, db, s.edge_alert_cooldown_minutes):
                continue
            label = m.get("outcome_label") or m.get("description") or token_id[:16]
            polymarket_url = f"https://polymarket.com/market/{token_id}"
            try:
                await telegram.alert(
                    market_label=label,
                    market_url=polymarket_url,
                    model_prob=model_prob,
                    pm_prob=pm_prob,
                    edge=edge,
                )
                # Mark alerted
                db.table("edges").update({
                    "alerted":    True,
                    "alerted_at": datetime.now(timezone.utc).isoformat(),
                }).eq("market_id", token_id).gte(
                    "computed_at",
                    (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
                ).execute()
                stats["alerted"] += 1
            except Exception as e:
                log.warning("edges.alert_failed", error=str(e))

    log.info("edges.recompute.done", **stats)
    return stats
