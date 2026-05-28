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


async def _tournament_prob(team_id: str, field: str, db) -> float | None:
    """Fetch a tournament probability (e.g. p_win_outright, p_win_group) for a team."""
    r = (db.table("tournament_predictions")
         .select(field)
         .eq("team_id", team_id)
         .order("computed_at", desc=True)
         .limit(1)
         .execute())
    if r.data and r.data[0].get(field) is not None:
        return float(r.data[0][field])
    return None


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



# ---------------------------------------------------------------------
# Vig / overround handling
# ---------------------------------------------------------------------

async def _event_overround(market_type: str, db) -> float:
    """
    Estimate total implied probability across all 'Yes' tokens of a multi-outcome
    event (outright winner across 48 teams, or one group's 4 teams).

    Polymarket prices don't sum to 1.0 — the excess is the market's overround
    (its margin). For outright markets this can be 1.10-1.30. We use it to
    compute a vig-adjusted 'fair' price so we don't flag the entire overround
    as fake negative edge.

    Returns the sum of latest prices across active markets of this type.
    Returns 1.0 if it can't be computed (no adjustment).
    """
    markets = (db.table("polymarket_markets")
               .select("id")
               .eq("active", True)
               .eq("market_type", market_type)
               .execute()).data or []
    if not markets:
        return 1.0

    total = 0.0
    counted = 0
    for mk in markets:
        pr = await _latest_price(mk["id"], db)
        if pr and pr.get("price") is not None:
            total += float(pr["price"])
            counted += 1
    # Only trust the overround if we have most of the field priced
    if counted < max(2, len(markets) // 2):
        return 1.0
    return total if total > 0 else 1.0


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
               .in_("market_type", ["match_1x2", "outright", "group_winner"])
               .execute())
    rows = markets.data or []

    stats = {"considered": len(rows), "edged": 0, "alerted": 0,
             "no_model": 0, "no_price": 0}

    # Precompute overround per multi-outcome market type so we can convert quoted
    # prices into no-vig "fair" prices for a cleaner edge signal.
    overrounds = {
        "outright":     await _event_overround("outright", db),
        "group_winner": await _event_overround("group_winner", db),
    }
    log.info("edges.overrounds", **{k: round(v, 3) for k, v in overrounds.items()})

    for m in rows:
        token_id = m["id"]

        # 1. Find latest price
        price_row = await _latest_price(token_id, db)
        if not price_row or price_row["price"] is None:
            stats["no_price"] += 1
            continue
        pm_prob = float(price_row["price"])

        # 2. Look up matching model prediction (varies by market type)
        team_id = m.get("team_id")
        if not team_id:
            stats["no_model"] += 1
            continue

        mtype = m.get("market_type")
        model_prob = None
        model_version = "unknown"

        if mtype == "outright":
            model_prob = await _tournament_prob(team_id, "p_win_outright", db)
            model_version = "tournament_sim"
        elif mtype == "group_winner":
            model_prob = await _tournament_prob(team_id, "p_win_group", db)
            model_version = "tournament_sim"
        elif mtype == "match_1x2":
            match_q = (db.table("matches")
                       .select("id,home_id,away_id,kickoff,competition")
                       .eq("completed", False)
                       .or_(f"home_id.eq.{team_id},away_id.eq.{team_id}")
                       .eq("competition", "WC2026")
                       .order("kickoff")
                       .limit(1)
                       .execute())
            match_rows = match_q.data or []
            if match_rows:
                match = match_rows[0]
                pred = await _latest_match_prediction(match["id"], db)
                if pred:
                    model_prob = (float(pred["p_home"])
                                  if match["home_id"] == team_id
                                  else float(pred["p_away"]))
                    model_version = pred["model_version"]

        if model_prob is None:
            stats["no_model"] += 1
            continue

        # 3. Compute edge. For multi-outcome markets, convert the quoted price
        #    to a no-vig "fair" price by dividing by the event overround, so we
        #    measure true mispricing rather than the market's built-in margin.
        overround = overrounds.get(mtype, 1.0)
        fair_pm_prob = pm_prob / overround if overround > 0 else pm_prob
        edge = model_prob - fair_pm_prob

        # 4. Always log it (store the fair price as pm_prob, raw kept in note)
        db.table("edges").insert({
            "market_id":     token_id,
            "model_prob":    model_prob,
            "pm_prob":       fair_pm_prob,
            "edge":          edge,
            "edge_lower_ci": edge - 0.05,
            "model_version": model_version,
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
                    pm_prob=fair_pm_prob,
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
