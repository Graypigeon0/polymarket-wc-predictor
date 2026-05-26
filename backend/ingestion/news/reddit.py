"""
Reddit poller.

Uses Reddit's public JSON endpoints (append `.json` to any URL).
No auth needed for read-only public subs. Polite rate limit + custom UA required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import structlog

log = structlog.get_logger()

SUBS = [
    "soccer",
    "worldcup",
    # National team subs added pre-tournament once 26-man squads land:
    # "BrazilFootball", "USMNT", "ThreeLions", "EquipeDeFrance", ...
]


@dataclass
class NewsItem:
    source: str
    url: str
    headline: str
    body: str
    published: datetime


async def fetch_all(limit: int = 25) -> list[NewsItem]:
    items: list[NewsItem] = []
    headers = {"User-Agent": "polymarket-wc-predictor/0.1 by /u/Graypigeon0"}
    async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
        for sub in SUBS:
            try:
                r = await client.get(
                    f"https://www.reddit.com/r/{sub}/hot.json?limit={limit}"
                )
                r.raise_for_status()
                for child in r.json().get("data", {}).get("children", []):
                    d = child.get("data", {})
                    items.append(NewsItem(
                        source=f"reddit:{sub}",
                        url=f"https://reddit.com{d.get('permalink', '')}",
                        headline=d.get("title", ""),
                        body=d.get("selftext", "") or d.get("url", ""),
                        published=datetime.fromtimestamp(
                            d.get("created_utc", 0), tz=timezone.utc
                        ),
                    ))
            except Exception as e:
                log.warning("reddit.fetch.error", sub=sub, error=str(e))
    log.info("reddit.fetched", count=len(items))
    return items
