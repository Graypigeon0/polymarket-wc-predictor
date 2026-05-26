"""
football-data.org ingestion (free tier).

Pulls fixtures + historical results from football-data.org's REST API.

Used for:
  - WC 2026 fixtures, group draws, kickoff times
  - Match results as they complete
  - Historical results for model fitting + backtesting:
      * Euros 2024, Euros 2020/21, Euro qualifiers
      * Copa America 2024, Copa America 2021
      * World Cup 2014, 2018, 2022 + WC qualifiers
      * UEFA Nations League (high-quality international matches)

Free tier: 10 req/min, generally covers competitions listed at:
https://www.football-data.org/coverage

Docs: https://www.football-data.org/documentation/quickstart
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings
from backend.db.client import get_client

log = structlog.get_logger()

BASE_URL = "https://api.football-data.org/v4"

# Mapping of internal competition codes to football-data.org competition codes
# and the relevant seasons to pull
COMPETITIONS: dict[str, dict[str, Any]] = {
    "WC2026":     {"code": "WC",  "seasons": [2026]},
    "WC2022":     {"code": "WC",  "seasons": [2022]},
    "WC2018":     {"code": "WC",  "seasons": [2018]},
    "WC2014":     {"code": "WC",  "seasons": [2014]},
    "EURO2024":   {"code": "EC",  "seasons": [2024]},
    "EURO2020":   {"code": "EC",  "seasons": [2020]},
    "NATIONS":    {"code": "UNL", "seasons": [2024, 2022]},
}


# ---------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------

def _headers() -> dict[str, str]:
    return {"X-Auth-Token": get_settings().football_data_api_key or ""}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
async def _get(client: httpx.AsyncClient, path: str, params: dict | None = None) -> dict:
    """GET with retry. Polite 6s sleep between calls (free tier = 10 req/min)."""
    r = await client.get(f"{BASE_URL}{path}", params=params,
                         headers=_headers(), timeout=30.0)
    r.raise_for_status()
    # Free-tier rate limit is 10 req/min, so we sleep AFTER each call
    await asyncio.sleep(6.5)
    return r.json()


# ---------------------------------------------------------------------
# Team upsert helpers
# ---------------------------------------------------------------------

# Map football-data.org country names to FIFA 3-letter codes for the
# teams most likely to appear at WC 2026. Extended on the fly as needed.
FIFA_CODES: dict[str, str] = {
    "Argentina": "ARG", "Australia": "AUS", "Austria": "AUT",
    "Belgium": "BEL", "Brazil": "BRA", "Cameroon": "CMR",
    "Canada": "CAN", "Chile": "CHI", "Colombia": "COL",
    "Costa Rica": "CRC", "Croatia": "CRO", "Czech Republic": "CZE",
    "Czechia": "CZE", "Denmark": "DEN", "Ecuador": "ECU",
    "Egypt": "EGY", "England": "ENG", "France": "FRA",
    "Germany": "GER", "Ghana": "GHA", "Hungary": "HUN",
    "Iran": "IRN", "Iran (Islamic Republic of)": "IRN",
    "Italy": "ITA", "Ivory Coast": "CIV", "Cote d'Ivoire": "CIV",
    "Japan": "JPN", "Korea Republic": "KOR", "South Korea": "KOR",
    "Mexico": "MEX", "Morocco": "MAR", "Netherlands": "NED",
    "Nigeria": "NGA", "Norway": "NOR", "Panama": "PAN",
    "Paraguay": "PAR", "Peru": "PER", "Poland": "POL",
    "Portugal": "POR", "Qatar": "QAT", "Romania": "ROU",
    "Saudi Arabia": "KSA", "Scotland": "SCO", "Senegal": "SEN",
    "Serbia": "SRB", "Slovakia": "SVK", "Slovenia": "SVN",
    "Spain": "ESP", "Sweden": "SWE", "Switzerland": "SUI",
    "Tunisia": "TUN", "Turkey": "TUR", "Tu\u0308rkiye": "TUR",
    "Ukraine": "UKR", "United States": "USA", "Uruguay": "URU",
    "Wales": "WAL",
}

# Confederation lookup (used only when creating a new team row)
CONFEDERATIONS: dict[str, str] = {
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Chile": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Panama": "CONCACAF",
    "Australia": "AFC", "Japan": "AFC", "Korea Republic": "AFC",
    "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC", "Qatar": "AFC",
    "Cameroon": "CAF", "Egypt": "CAF", "Ghana": "CAF", "Ivory Coast": "CAF",
    "Cote d'Ivoire": "CAF", "Morocco": "CAF", "Nigeria": "CAF",
    "Senegal": "CAF", "Tunisia": "CAF",
}


def _fifa_code(country: str) -> str:
    """Lookup or synthesise a 3-letter code for unknown teams."""
    if country in FIFA_CODES:
        return FIFA_CODES[country]
    # Fallback: take first 3 uppercase letters
    cleaned = "".join(c for c in country if c.isalpha()).upper()
    return cleaned[:3] if cleaned else "UNK"


def _confederation(country: str) -> str:
    return CONFEDERATIONS.get(country, "UEFA")  # default to UEFA — most common


async def _upsert_team(country: str) -> str:
    """Ensure a team row exists for `country`. Returns the team UUID."""
    db = get_client()
    code = _fifa_code(country)

    existing = db.table("teams").select("id").eq("fifa_code", code).execute()
    if existing.data:
        return existing.data[0]["id"]

    team_id = str(uuid.uuid4())
    db.table("teams").insert({
        "id": team_id,
        "fifa_code": code,
        "name": country,
        "confederation": _confederation(country),
    }).execute()
    log.info("football_data.team_created", country=country, code=code)
    return team_id


# ---------------------------------------------------------------------
# Match upsert
# ---------------------------------------------------------------------

# football-data.org stage strings → our internal stage codes
STAGE_MAP: dict[str, str] = {
    "GROUP_STAGE":          "group",
    "LAST_16":              "r16",
    "ROUND_OF_16":          "r16",
    "QUARTER_FINALS":       "qf",
    "SEMI_FINALS":          "sf",
    "THIRD_PLACE":          "3p",
    "FINAL":                "final",
    "PRELIMINARY_ROUND":    "qual",
    "QUALIFICATION":        "qual",
    "REGULAR_SEASON":       "qual",
    "LEAGUE_STAGE":         "qual",
}


async def _upsert_match(m: dict, competition_code: str) -> None:
    """Upsert a single match row from a football-data.org match payload."""
    db = get_client()

    home_country = m["homeTeam"]["name"]
    away_country = m["awayTeam"]["name"]
    home_id = await _upsert_team(home_country)
    away_id = await _upsert_team(away_country)

    # Score (None until match is finished)
    home_goals = None
    away_goals = None
    score = m.get("score", {}).get("fullTime") or {}
    if score.get("home") is not None and score.get("away") is not None:
        home_goals = int(score["home"])
        away_goals = int(score["away"])

    stage = STAGE_MAP.get(m.get("stage", "GROUP_STAGE"), "group")
    completed = m.get("status") in ("FINISHED", "AWARDED")
    venue = m.get("venue") or ""

    # Use football-data match id as a stable external key in `id`
    # We use a deterministic UUID5 from competition + their match id so
    # the upsert is idempotent across runs.
    external_key = f"footballdata:{competition_code}:{m['id']}"
    match_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, external_key))

    row = {
        "id":          match_uuid,
        "home_id":     home_id,
        "away_id":     away_id,
        "kickoff":     m["utcDate"],
        "venue":       venue,
        "stage":       stage,
        "competition": competition_code,
        "is_neutral":  competition_code.startswith(("WC", "EURO", "COPA")),
        "home_goals":  home_goals,
        "away_goals":  away_goals,
        "completed":   completed,
    }
    db.table("matches").upsert(row, on_conflict="id").execute()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

async def refresh_competition(competition_code: str) -> int:
    """
    Pull all matches for one competition (across all configured seasons)
    and upsert them. Returns the count of matches written.
    """
    conf = COMPETITIONS.get(competition_code)
    if not conf:
        log.error("football_data.unknown_competition", code=competition_code)
        return 0

    written = 0
    async with httpx.AsyncClient() as client:
        for season in conf["seasons"]:
            try:
                log.info("football_data.fetch", code=competition_code, season=season)
                data = await _get(
                    client,
                    f"/competitions/{conf['code']}/matches",
                    {"season": season},
                )
                matches = data.get("matches", [])
                for m in matches:
                    try:
                        await _upsert_match(m, competition_code)
                        written += 1
                    except Exception as e:
                        log.warning("football_data.match_upsert_failed",
                                    match_id=m.get("id"), error=str(e))
            except httpx.HTTPStatusError as e:
                # 403 commonly means "this competition/season not on your tier"
                log.warning("football_data.competition_skipped",
                            code=competition_code, season=season,
                            status=e.response.status_code)
            except Exception as e:
                log.error("football_data.fetch_failed",
                          code=competition_code, season=season, error=str(e))

    log.info("football_data.competition_done",
             code=competition_code, written=written)
    return written


async def refresh_fixtures() -> int:
    """
    Refresh all configured competitions.

    Called from backend.main on the squad-refresh schedule.
    Returns total matches upserted across all competitions.
    """
    log.info("football_data.refresh.start")
    total = 0
    for code in COMPETITIONS:
        total += await refresh_competition(code)
    log.info("football_data.refresh.done", total=total)
    return total
