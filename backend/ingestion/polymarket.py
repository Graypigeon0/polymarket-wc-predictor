"""
Polymarket ingestion.

Two APIs:
  - Gamma:  https://gamma-api.polymarket.com   (discovery, public, no auth)
  - CLOB:   https://clob.polymarket.com         (orderbook + prices, public read)

We use:
  /events    on Gamma — World Cup 2026 events (groups markets together)
  /markets   on Gamma — individual markets with token IDs
  /book      on CLOB  — orderbook midpoint + depth per token
  /midpoint  on CLOB  — fast midpoint price
"""

from __future__ import annotations

import re
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.db.client import get_client

log = structlog.get_logger()

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

# Tags / keywords that identify WC 2026 markets
WC_KEYWORDS = ["world cup 2026", "fifa world cup", "wc 2026"]


# ---------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _gamma_get(client: httpx.AsyncClient, path: str, params: dict | None = None) -> Any:
    r = await client.get(f"{GAMMA_BASE}{path}", params=params, timeout=20.0)
    r.raise_for_status()
    return r.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _clob_get(client: httpx.AsyncClient, path: str, params: dict | None = None) -> Any:
    r = await client.get(f"{CLOB_BASE}{path}", params=params, timeout=20.0)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------
# Market classification
# ---------------------------------------------------------------------

def _classify_market(question: str, slug: str = "") -> str:
    """
    Classify a Polymarket market by its question text.
    Returns one of: match_1x2, outright, group_winner, stage_advance,
                    top_scorer, golden_boot, other.
    """
    q = (question or "").lower()
    s = (slug or "").lower()
    text = f"{q} {s}"

    if any(k in text for k in ["win the 2026 world cup", "win the world cup",
                                "world cup winner", "win wc 2026", "win wc26"]):
        return "outright"
    if "top scorer" in text or "golden boot" in text:
        return "top_scorer"
    if "win group" in text or "winner of group" in text:
        return "group_winner"
    if any(k in text for k in ["reach the final", "make the final",
                                "reach the semi", "reach the quarter",
                                "advance from group", "make it out of group"]):
        return "stage_advance"
    # Match 1x2 detection: "X vs Y" or "X to beat Y"
    if re.search(r"\b\w+\s+vs\.?\s+\w+\b", text) or " to beat " in text:
        return "match_1x2"
    return "other"


def _extract_teams(question: str) -> list[str]:
    """Extract team names from a market question. Returns up to 2."""
    q = question or ""
    # "Argentina vs Mexico" → ["Argentina", "Mexico"]
    m = re.search(r"([A-Z][a-zA-Z\s]+?)\s+vs\.?\s+([A-Z][a-zA-Z\s]+?)[\?\:\,\.\s]",
                  q + " ")
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    # "X to beat Y"
    m = re.search(r"([A-Z][a-zA-Z\s]+?)\s+to beat\s+([A-Z][a-zA-Z\s]+)", q)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    return []


async def _lookup_team_id(name: str) -> str | None:
    """Find our internal team UUID by name (case-insensitive, partial match)."""
    if not name:
        return None
    db = get_client()
    # Exact name match first
    r = db.table("teams").select("id").ilike("name", name).execute()
    if r.data:
        return r.data[0]["id"]
    # Try as FIFA code
    if len(name) == 3:
        r = db.table("teams").select("id").eq("fifa_code", name.upper()).execute()
        if r.data:
            return r.data[0]["id"]
    return None


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------

async def discover_wc_markets() -> dict[str, int]:
    """
    Find all active WC 2026 markets on Polymarket via the /events endpoint and
    upsert into polymarket_markets. Returns counts by market_type.

    Polymarket groups related markets into "events". A World Cup winner event
    contains ~48 sub-markets, one per team, each a binary Yes/No with the team
    name in `groupItemTitle`. We classify the EVENT (outright / group_winner /
    top_scorer) then attach each sub-market's team.
    """
    log.info("polymarket.discover.start")
    counts: dict[str, int] = {}
    db = get_client()

    # WC 2026 event slugs we care about. Gamma supports tag/slug search; we
    # query broadly then filter by title keywords.
    async with httpx.AsyncClient() as client:
        events = await _find_wc_events(client)
        log.info("polymarket.discover.events_found", n=len(events))

        for ev in events:
            ev_title = (ev.get("title") or "").lower()
            ev_slug = (ev.get("slug") or "").lower()
            market_type = _classify_event(ev_title, ev_slug)

            for m in ev.get("markets", []):
                if m.get("closed") or m.get("archived"):
                    continue
                token_ids = _parse_json_field(m.get("clobTokenIds"))
                outcomes = _parse_json_field(m.get("outcomes")) or ["Yes", "No"]
                if not token_ids:
                    continue

                # Team name: neg-risk events put it in groupItemTitle
                team_name = (m.get("groupItemTitle") or "").strip()
                if not team_name:
                    # Fall back to extracting from the question
                    names = _extract_teams(m.get("question", ""))
                    team_name = names[0] if names else ""

                team_id = await _lookup_team_id(team_name) if team_name else None
                counts[market_type] = counts.get(market_type, 0) + 1

                # For binary team markets we track the "Yes" token (index 0).
                yes_token = str(token_ids[0])
                label_team = team_name or m.get("question", "")[:40]
                db.table("polymarket_markets").upsert({
                    "id":            yes_token,
                    "market_type":   market_type,
                    "description":   m.get("question") or ev.get("title"),
                    "outcome_label": f"{ev.get('title', '')} — {label_team}",
                    "team_id":       team_id,
                    "event_slug":    ev.get("slug"),
                    "active":        True,
                }, on_conflict="id").execute()

    log.info("polymarket.discover.done", by_type=counts,
             total=sum(counts.values()))
    return counts


async def _find_wc_events(client: httpx.AsyncClient) -> list[dict]:
    """Search Gamma /events for active World Cup 2026 events (paginated)."""
    found: list[dict] = []
    seen_ids: set = set()
    offset = 0
    page_size = 100

    while True:
        try:
            data = await _gamma_get(client, "/events", params={
                "active": "true",
                "closed": "false",
                "archived": "false",
                "limit": str(page_size),
                "offset": str(offset),
            })
        except Exception as e:
            log.error("polymarket.events_failed", error=str(e), offset=offset)
            break

        events = data if isinstance(data, list) else data.get("data", [])
        if not events:
            break

        for ev in events:
            text = ((ev.get("title") or "") + " " + (ev.get("slug") or "")).lower()
            if any(k in text for k in WC_KEYWORDS):
                if ev.get("id") not in seen_ids:
                    seen_ids.add(ev.get("id"))
                    found.append(ev)

        if len(events) < page_size:
            break
        offset += page_size
        if offset > 5000:   # safety cap
            break

    return found


def _classify_event(title: str, slug: str) -> str:
    """Classify a Polymarket EVENT (not individual market) by title/slug."""
    text = f"{title} {slug}".lower()
    if "top scorer" in text or "golden boot" in text:
        return "top_scorer"
    if "win group" in text or "group" in text and "winner" in text:
        return "group_winner"
    if any(k in text for k in ["winner", "win the", "to win", "champion"]):
        return "outright"
    if "reach" in text or "advance" in text or "make the" in text:
        return "stage_advance"
    return "other"


def _parse_json_field(val) -> list:
    """Gamma sometimes returns arrays as JSON strings. Parse defensively."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        import json as _json
        try:
            parsed = _json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


async def refresh_prices() -> int:
    """
    Pull latest prices for all tracked WC markets via the Gamma API.

    We re-scan the WC events (same path used for discovery, known reachable from
    CI) and read each market's current price directly from the Gamma market
    object — `bestBid`/`bestAsk` midpoint, falling back to `outcomePrices[0]`
    (the Yes price) or `lastTradePrice`. This avoids the CLOB /book endpoint,
    which is geo/IP-restricted and unreachable from GitHub Actions runners.
    """
    log.info("polymarket.refresh_prices.start")
    db = get_client()

    # Which token_ids are we tracking?
    tracked = (db.table("polymarket_markets")
               .select("id")
               .eq("active", True)
               .execute()).data or []
    tracked_ids = {r["id"] for r in tracked}
    if not tracked_ids:
        log.info("polymarket.refresh_prices.no_markets")
        return 0

    written = 0
    async with httpx.AsyncClient() as client:
        events = await _find_wc_events(client)
        for ev in events:
            for m in ev.get("markets", []):
                token_ids = _parse_json_field(m.get("clobTokenIds"))
                if not token_ids:
                    continue
                yes_token = str(token_ids[0])
                if yes_token not in tracked_ids:
                    continue

                price = _extract_price(m)
                if price is None:
                    continue

                best_bid = _safe_float(m.get("bestBid"))
                best_ask = _safe_float(m.get("bestAsk"))
                # Liquidity proxy from Gamma
                depth = _safe_float(m.get("liquidityNum")) or _safe_float(m.get("liquidity")) or 0.0

                db.table("polymarket_prices").insert({
                    "market_id":  yes_token,
                    "price":      price,
                    "bid":        best_bid,
                    "ask":        best_ask,
                    "book_depth": depth,
                }).execute()
                written += 1

    log.info("polymarket.refresh_prices.done", written=written,
             tracked=len(tracked_ids))
    return written


def _safe_float(val) -> float | None:
    try:
        if val is None or val == "":
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_price(market: dict) -> float | None:
    """Get the current Yes-price for a Gamma market object."""
    # 1. Midpoint of best bid/ask if both present
    bid = _safe_float(market.get("bestBid"))
    ask = _safe_float(market.get("bestAsk"))
    if bid is not None and ask is not None and 0 < bid <= ask <= 1:
        return (bid + ask) / 2.0
    # 2. outcomePrices[0] (the Yes price)
    prices = _parse_json_field(market.get("outcomePrices"))
    if prices:
        p = _safe_float(prices[0])
        if p is not None and 0 <= p <= 1:
            return p
    # 3. last trade
    ltp = _safe_float(market.get("lastTradePrice"))
    if ltp is not None and 0 <= ltp <= 1:
        return ltp
    return None
