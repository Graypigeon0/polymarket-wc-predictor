"""
FBref ingestion (fallback + complement to Sofascore).

Scrapes club-season underlying stats from fbref.com:
  - xG, xA, shots per 90
  - Defensive actions, tackle %, GK saves %
  - Used as quality baseline + Sofascore-blocked fallback.

FBref serves static HTML — much more reliable than scraping JS apps.
Polite rate limit: 1 req every 3 seconds.
"""

from __future__ import annotations

import asyncio

import httpx
import structlog
from selectolax.parser import HTMLParser

log = structlog.get_logger()

BASE_URL = "https://fbref.com"


async def refresh_underlying_stats() -> None:
    """
    Pull xG / xGA / shots per 90 etc. for all WC 26-man squad players.

    Plan:
      1. Read called-up players from `squads` (with club info).
      2. Fetch club's most recent league season standard stats page.
      3. Parse player rows, upsert into `players` (club_xg90, club_xga90, etc.).
    """
    log.info("fbref.refresh.todo")
    # TODO: implement


async def _fetch_html(client: httpx.AsyncClient, path: str) -> HTMLParser:
    await asyncio.sleep(3.0)  # FBref asks for max 1 req every 3s
    r = await client.get(f"{BASE_URL}{path}", timeout=20.0,
                         headers={"User-Agent": "polymarket-wc-predictor/0.1"})
    r.raise_for_status()
    return HTMLParser(r.text)
