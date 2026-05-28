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
    Find all active WC 2026 markets on Polymarket and upsert into the DB.
    Returns counts by market_type.
    """
    log.info("polymarket.discover.start")
    counts: dict[str, int] = {}
    db = get_client()

    async with httpx.AsyncClient() as client:
        # Strategy: paginate /markets with active=true and filter by tag/question
        # Gamma supports up to 500 markets per page.
        offset = 0
        page_size = 500
        seen = 0
        matched = 0

        while True:
            try:
                params = {
                    "active": "true",
                    "closed": "false",
                    "archived": "false",
                    "limit": str(page_size),
                    "offset": str(offset),
                }
                data = await _gamma_get(client, "/markets", params=params)
            except Exception as e:
                log.error("polymarket.gamma_failed", error=str(e))
                break

            markets = data if isinstance(data, list) else data.get("data", [])
            if not markets:
                break
            seen += len(markets)

            for m in markets:
                question = m.get("question", "")
                slug = m.get("slug", "")
                tags = m.get("tags") or []
                text = (question + " " + slug + " " + " ".join(tags)).lower()

                # Filter to WC 2026 only
                if not any(k in text for k in WC_KEYWORDS):
                    continue

                condition_id = m.get("conditionId") or m.get("condition_id")
                clob_token_ids = m.get("clobTokenIds") or m.get("clob_token_ids") or []
                if isinstance(clob_token_ids, str):
                    # Sometimes returned as JSON string
                    import json as _json
                    try:
                        clob_token_ids = _json.loads(clob_token_ids)
                    except Exception:
                        clob_token_ids = []
                if not condition_id or not clob_token_ids:
                    continue

                market_type = _classify_market(question, slug)
                counts[market_type] = counts.get(market_type, 0) + 1

                # Try to extract team(s) for matching to internal data
                team_ids: list[str | None] = []
                if market_type in ("match_1x2", "outright", "group_winner",
                                   "stage_advance"):
                    names = _extract_teams(question)
                    # Also check for outright "Brazil to win" format
                    if not names:
                        m2 = re.search(r"([A-Z][a-zA-Z\s]+?)\s+to win",
                                       question or "")
                        if m2:
                            names = [m2.group(1).strip()]
                    for n in names[:2]:
                        team_ids.append(await _lookup_team_id(n))

                # Each market has 2 outcome tokens (typically Yes/No). For binary
                # win markets, we just track the "Yes" token (index 0).
                outcomes = m.get("outcomes") or ["Yes", "No"]
                if isinstance(outcomes, str):
                    import json as _json
                    try:
                        outcomes = _json.loads(outcomes)
                    except Exception:
                        outcomes = ["Yes", "No"]

                for idx, token_id in enumerate(clob_token_ids[:2]):
                    if not token_id:
                        continue
                    label = outcomes[idx] if idx < len(outcomes) else f"Outcome {idx}"

                    row = {
                        "id":            str(token_id),
                        "market_type":   market_type,
                        "description":   question,
                        "outcome_label": f"{question} — {label}",
                        "active":        True,
                    }
                    # Attach team_id when we extracted one
                    if team_ids:
                        if market_type == "match_1x2" and len(team_ids) >= 2:
                            # idx 0 = first team wins; idx 1 = second team wins
                            row["team_id"] = team_ids[idx] if idx < len(team_ids) else None
                        elif market_type in ("outright", "group_winner",
                                             "stage_advance"):
                            # "Brazil to win" — same team on both Yes/No tokens
                            row["team_id"] = team_ids[0]

                    db.table("polymarket_markets").upsert(
                        row, on_conflict="id").execute()
                    matched += 1

            if len(markets) < page_size:
                break
            offset += page_size

    log.info("polymarket.discover.done",
             total_scanned=seen, wc_matched=matched, by_type=counts)
    return counts


# ---------------------------------------------------------------------
# Price refresh
# ---------------------------------------------------------------------

async def refresh_prices() -> int:
    """Pull latest midpoint+book for every active polymarket_markets row."""
    log.info("polymarket.refresh_prices.start")
    db = get_client()

    markets = (db.table("polymarket_markets")
               .select("id")
               .eq("active", True)
               .execute())
    market_ids = [r["id"] for r in (markets.data or [])]
    if not market_ids:
        log.info("polymarket.refresh_prices.no_markets")
        return 0

    written = 0
    async with httpx.AsyncClient() as client:
        for token_id in market_ids:
            try:
                # Use /midpoint for speed (one request gives midpoint + best bid/ask)
                book = await _clob_get(client, "/book", params={"token_id": token_id})
                bids = book.get("bids") or []
                asks = book.get("asks") or []
                best_bid = float(bids[0]["price"]) if bids else None
                best_ask = float(asks[0]["price"]) if asks else None

                if best_bid is not None and best_ask is not None:
                    price = (best_bid + best_ask) / 2.0
                elif best_bid is not None:
                    price = best_bid
                elif best_ask is not None:
                    price = best_ask
                else:
                    continue  # no liquidity, skip

                # Rough order-book depth within 1pp of midpoint
                depth = 0.0
                for side in (bids, asks):
                    for level in side:
                        try:
                            p = float(level["price"])
                            sz = float(level.get("size", 0))
                            if abs(p - price) <= 0.01:
                                depth += sz * p
                        except (KeyError, TypeError, ValueError):
                            continue

                db.table("polymarket_prices").insert({
                    "market_id":  token_id,
                    "price":      price,
                    "bid":        best_bid,
                    "ask":        best_ask,
                    "book_depth": depth,
                }).execute()
                written += 1
            except Exception as e:
                log.warning("polymarket.price_failed",
                            token_id=token_id[:16] + "...", error=str(e)[:200])

    log.info("polymarket.refresh_prices.done", written=written)
    return written
