"""
martj42/international_results CSV ingestion.

Single canonical source for international match results since 1872, including
friendlies, qualifiers, and minor tournaments. Free, public domain, no API key.

Source: https://github.com/martj42/international_results
Raw URL: https://raw.githubusercontent.com/martj42/international_results/master/results.csv

CSV columns:
  date         YYYY-MM-DD
  home_team    e.g. "Brazil"
  away_team    e.g. "Argentina"
  home_score   int
  away_score   int
  tournament   e.g. "FIFA World Cup", "Friendly", "UEFA Nations League", ...
  city         e.g. "Doha"
  country      country where the match was played
  neutral      "True" / "False"

We filter to last N years (default 5) to keep training data relevant to
current squad compositions and tactical eras.
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import date, datetime, timedelta, timezone

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.db.client import get_client
from backend.ingestion.football_data import _confederation, _fifa_code

log = structlog.get_logger()

CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# How many years back to pull. 5 years captures current squad eras while
# keeping the dataset manageable (~5000 matches).
LOOKBACK_YEARS = 5


# ---------------------------------------------------------------------
# Tournament → internal competition code mapping
# ---------------------------------------------------------------------

# Maps lowercase tournament-name substrings to our competition codes.
# Order matters — first match wins. Specific competitions first, then qualifiers,
# then catchalls.
TOURNAMENT_MAP: list[tuple[str, str]] = [
    # World Cup main tournaments (handled by openfootball; this is a safety net)
    ("fifa world cup qualification", "WCQUAL"),
    ("fifa world cup",                "WCMAIN"),   # shouldn't reach here for 14/18/22

    # Continental championships
    ("uefa euro qualification",       "EURQUAL"),
    ("uefa european championship qualification", "EURQUAL"),
    ("uefa euro",                     "EURMAIN"),  # finals handled elsewhere
    ("uefa european championship",    "EURMAIN"),

    # Nations League (UEFA + others use the same format)
    ("uefa nations league",           "UNL"),
    ("concacaf nations league",       "CNL"),

    # Other continental cups
    ("copa américa",                  "COPA"),
    ("copa america",                  "COPA"),
    ("africa cup of nations",         "AFCON"),
    ("african cup of nations",        "AFCON"),
    ("afc asian cup",                 "ASIAN"),
    ("oceania nations cup",           "OFC"),

    # Continental qualifiers
    ("afc asian cup qualification",   "ASIANQUAL"),
    ("africa cup of nations qualification", "AFCONQUAL"),

    # Confederation cups
    ("concacaf gold cup",             "GOLDCUP"),
    ("concacaf championship",         "GOLDCUP"),

    # Generic
    ("friendly",                      "FRIENDLY"),
]


def _competition_for(tournament: str) -> str:
    t = (tournament or "").lower().strip()
    for substr, code in TOURNAMENT_MAP:
        if substr in t:
            return code
    return "OTHER"


# ---------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
async def _fetch_csv() -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(CSV_URL)
        r.raise_for_status()
        return r.text


# ---------------------------------------------------------------------
# Team upsert (uses football_data helpers)
# ---------------------------------------------------------------------

# Cache team_id lookups within a single run to cut Supabase round-trips.
_team_cache: dict[str, str] = {}


async def _upsert_team(country: str) -> str | None:
    if not country:
        return None
    if country in _team_cache:
        return _team_cache[country]

    db = get_client()
    code = _fifa_code(country)

    existing = db.table("teams").select("id").eq("fifa_code", code).execute()
    if existing.data:
        team_id = existing.data[0]["id"]
    else:
        team_id = str(uuid.uuid4())
        db.table("teams").insert({
            "id": team_id,
            "fifa_code": code,
            "name": country,
            "confederation": _confederation(country),
        }).execute()

    _team_cache[country] = team_id
    return team_id


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

async def refresh(lookback_years: int = LOOKBACK_YEARS) -> int:
    """
    Pull the martj42 results CSV, filter to recent matches, upsert into `matches`.
    Returns the number of matches written.
    """
    log.info("martj42.refresh.start", lookback_years=lookback_years)
    text = await _fetch_csv()
    reader = csv.DictReader(io.StringIO(text))

    cutoff = date.today() - timedelta(days=365 * lookback_years)
    db = get_client()
    written = 0
    skipped = 0

    # Batch upserts in chunks of 500 to cut HTTP round-trips
    batch: list[dict] = []
    BATCH_SIZE = 500

    async def _flush() -> None:
        nonlocal batch
        if not batch:
            return
        try:
            db.table("matches").upsert(batch, on_conflict="id").execute()
        except Exception as e:
            log.warning("martj42.batch_failed", size=len(batch), error=str(e)[:200])
        batch = []

    for row in reader:
        try:
            d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            if d < cutoff:
                continue

            home = row.get("home_team", "").strip()
            away = row.get("away_team", "").strip()
            hs   = row.get("home_score", "").strip()
            as_  = row.get("away_score", "").strip()
            if not home or not away or not hs or not as_:
                skipped += 1
                continue

            home_id = await _upsert_team(home)
            away_id = await _upsert_team(away)
            if not home_id or not away_id:
                skipped += 1
                continue

            competition = _competition_for(row.get("tournament", ""))
            neutral = (row.get("neutral", "").strip().lower() == "true")
            kickoff = f"{row['date']}T00:00:00Z"

            external_key = f"martj42:{home}-{away}-{row['date']}"
            match_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, external_key))

            batch.append({
                "id":          match_uuid,
                "home_id":     home_id,
                "away_id":     away_id,
                "kickoff":     kickoff,
                "venue":       row.get("city") or "",
                "stage":       "qual" if "QUAL" in competition else (
                               "group" if competition.startswith(("WC", "EUR", "COPA",
                                                                  "AFCON", "ASIAN", "GOLD"))
                               else "friendly" if competition == "FRIENDLY" else "group"),
                "competition": competition,
                "is_neutral":  neutral,
                "home_goals":  int(hs),
                "away_goals":  int(as_),
                "completed":   True,
            })

            if len(batch) >= BATCH_SIZE:
                await _flush()

            written += 1
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                log.warning("martj42.row_failed", error=str(e)[:200],
                            home=row.get("home_team"), away=row.get("away_team"))

    await _flush()
    log.info("martj42.refresh.done", written=written, skipped=skipped,
             cached_teams=len(_team_cache))
    return written
