"""
football-data.org ingestion (free tier).

Used for:
  - WC 2026 fixtures, group draws, kickoff times
  - Final results once matches complete
  - Historical results for backtest (Euros 2024, Copa 2024, prior WCs)

Free tier: 10 req/min. Plenty for fixtures.
Docs: https://www.football-data.org/documentation/quickstart
"""

from __future__ import annotations

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings

log = structlog.get_logger()

BASE_URL = "https://api.football-data.org/v4"

# Competition codes
COMPETITIONS = {
    "WC2026": "WC",          # FIFA World Cup
    "EURO2024": "EC",        # UEFA Euro
    "COPA2024": "CA",        # Copa America (may not be in free tier)
}


def _headers() -> dict[str, str]:
    return {"X-Auth-Token": get_settings().football_data_api_key or ""}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
async def _get(path: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}{path}", params=params,
                             headers=_headers(), timeout=20.0)
        r.raise_for_status()
        return r.json()


async def refresh_fixtures() -> None:
    """Pull all WC 2026 fixtures and upsert into `matches`."""
    log.info("football_data.fixtures.todo")
    # TODO: data = await _get("/competitions/WC/matches", {"season": 2026})
    # parse data["matches"] and upsert
