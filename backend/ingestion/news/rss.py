"""RSS news poller for major football outlets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import feedparser
import httpx
import structlog

log = structlog.get_logger()

FEEDS = {
    "bbc":      "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "guardian": "https://www.theguardian.com/football/rss",
    "espn":     "https://www.espn.com/espn/rss/soccer/news",
    # Add national team-specific feeds during tournament as needed
}


@dataclass
class NewsItem:
    source: str
    url: str
    headline: str
    body: str
    published: datetime


async def fetch_all() -> list[NewsItem]:
    items: list[NewsItem] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for source, url in FEEDS.items():
            try:
                r = await client.get(url)
                r.raise_for_status()
                parsed = feedparser.parse(r.text)
                for entry in parsed.entries[:30]:
                    items.append(NewsItem(
                        source=source,
                        url=entry.get("link", ""),
                        headline=entry.get("title", ""),
                        body=entry.get("summary", ""),
                        published=datetime.now(),  # TODO: parse entry.published
                    ))
            except Exception as e:
                log.warning("rss.fetch.error", source=source, error=str(e))
    log.info("rss.fetched", count=len(items))
    return items
