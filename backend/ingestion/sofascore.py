"""
Sofascore ingestion.

Scrapes Sofascore's undocumented public API for:
  - Player profiles, recent ratings (0-10 per match), minutes
  - Team rosters / national team callups

NOTE: No official API. Be polite: 1 req/sec with jitter, rotating user-agents,
exponential backoff on 429. If they tighten access, fall back to FBref.

Public endpoints (subject to change):
  /api/v1/team/{team_id}/players
  /api/v1/player/{player_id}/statistics/{season_id}/{tournament_id}
  /api/v1/national-team-suggested-players/{team_id}
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger()

BASE_URL = "https://api.sofascore.com"
USER_AGENTS = [
    # rotate a small pool of realistic UAs
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]


def _headers() -> dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
async def _get(client: httpx.AsyncClient, path: str) -> dict[str, Any]:
    await asyncio.sleep(1.0 + random.random())  # polite rate limit
    r = await client.get(f"{BASE_URL}{path}", headers=_headers(), timeout=20.0)
    if r.status_code == 429:
        raise httpx.HTTPError("rate limited")
    r.raise_for_status()
    return r.json()


async def refresh_players() -> None:
    """
    Refresh Sofascore data for all WC 2026 squad members.

    Plan:
      1. Read called-up players from `squads` table.
      2. For each player, fetch latest stats + rating.
      3. Upsert into `players` table.
    """
    log.info("sofascore.refresh_players.todo")
    # TODO: implement once `squads` is populated post-June 1 deadline


async def fetch_player_stats(sofascore_id: int) -> dict[str, Any]:
    """Fetch a single player's recent stats."""
    async with httpx.AsyncClient() as client:
        return await _get(client, f"/api/v1/player/{sofascore_id}")
