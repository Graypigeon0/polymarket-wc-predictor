"""
Polymarket CLOB client.

Public, free, no auth required for reading markets and prices.
We only read; never trade from this app.

Docs: https://docs.polymarket.com/
Useful endpoints:
  GET /markets                      list markets (paginated)
  GET /markets/{condition_id}       single market
  GET /book?token_id=...            order book for an outcome token
  GET /price?token_id=...&side=BUY  current best price
"""

from __future__ import annotations

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings

log = structlog.get_logger()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _get(client: httpx.AsyncClient, path: str, params: dict | None = None) -> dict:
    base = get_settings().polymarket_clob_url
    r = await client.get(f"{base}{path}", params=params, timeout=20.0)
    r.raise_for_status()
    return r.json()


async def discover_wc_markets() -> None:
    """
    One-shot: scan all active markets, identify WC 2026 ones, classify by type
    (match_1x2 / exact_score / outright / top_scorer), upsert into
    `polymarket_markets`. Run manually before each tournament phase.
    """
    log.info("polymarket.discover.todo")
    # TODO: paginate /markets, filter by tag / title containing "World Cup 2026"
    # TODO: map each market to internal team_id / match_id / player_id


async def refresh_prices() -> None:
    """
    For every active row in `polymarket_markets`, pull latest order book midpoint
    and depth, upsert into `polymarket_prices`.
    """
    log.info("polymarket.refresh.todo")
    # TODO: for each market token, GET /book, compute midpoint + depth-within-1pp
