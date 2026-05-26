"""
openfootball ingestion — free historical match data from GitHub.

openfootball is a crowdsourced repo of football data in clean JSON format,
no API key, no rate limit, no paywall. We use it for historical backtest data:
  - World Cups 2014, 2018, 2022
  - Euros 2020/21, 2024
  - Copa America 2024
  - International friendlies + UNL where useful

Source: https://github.com/openfootball
Raw URL pattern: https://raw.githubusercontent.com/openfootball/world-cup.json/master/<season>/worldcup.json
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.db.client import get_client
from backend.ingestion.football_data import _confederation, _fifa_code

log = structlog.get_logger()

# Map our competition codes to openfootball JSON URLs.
# openfootball uses different repos for different competitions; we hand-pick
# the most reliable ones below.
SOURCES: dict[str, str] = {
    "WC2022": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json",
    "WC2018": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2018/worldcup.json",
    "WC2014": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2014/worldcup.json",
    "EURO2024": "https://raw.githubusercontent.com/openfootball/euro.json/master/2024/euro.json",
    "EURO2020": "https://raw.githubusercontent.com/openfootball/euro.json/master/2020/euro.json",
}

STAGE_MAP: dict[str, str] = {
    "Group A":            "group",  "Group B":  "group",
    "Group C":            "group",  "Group D":  "group",
    "Group E":            "group",  "Group F":  "group",
    "Group G":            "group",  "Group H":  "group",
    "Round of 16":        "r16",
    "Quarter-finals":     "qf",     "Quarterfinals":    "qf",
    "Semi-finals":        "sf",     "Semifinals":       "sf",
    "Third-place play-off": "3p",   "Match for third place": "3p",
    "Final":              "final",
}


# ---------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
async def _fetch_json(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    r = await client.get(url, timeout=30.0)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------
# Team helpers (reuse football_data lookups for consistency)
# ---------------------------------------------------------------------

async def _upsert_team(country: str) -> str:
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
    return team_id


# ---------------------------------------------------------------------
# Match parsing
# ---------------------------------------------------------------------

def _parse_score(match: dict) -> tuple[int | None, int | None]:
    """openfootball stores score as 'score' dict or {'ft': [h, a]} or {'score1', 'score2'}."""
    if "score" in match and isinstance(match["score"], dict):
        ft = match["score"].get("ft") or match["score"].get("fullTime")
        if isinstance(ft, list) and len(ft) == 2:
            return int(ft[0]), int(ft[1])
    if "score1" in match and "score2" in match:
        try:
            return int(match["score1"]), int(match["score2"])
        except (TypeError, ValueError):
            return None, None
    return None, None


def _team_name(team_field) -> str | None:
    """openfootball team field can be a string or {'name': ..., 'code': ...}."""
    if isinstance(team_field, str):
        return team_field
    if isinstance(team_field, dict):
        return team_field.get("name") or team_field.get("country")
    return None


async def _upsert_match(match: dict, competition_code: str, stage_default: str) -> bool:
    """Upsert a single match from openfootball JSON. Returns True on success."""
    db = get_client()

    home_name = _team_name(match.get("team1"))
    away_name = _team_name(match.get("team2"))
    if not home_name or not away_name:
        return False

    home_id = await _upsert_team(home_name)
    away_id = await _upsert_team(away_name)
    home_goals, away_goals = _parse_score(match)

    # Stage may be on the match or inherited from the parent round
    stage = STAGE_MAP.get(match.get("group", stage_default), stage_default)

    # Build a deterministic UUID per match
    kickoff = match.get("date") or match.get("utcDate") or ""
    if "time" in match:
        kickoff = f"{kickoff}T{match['time']}:00Z"
    elif kickoff and "T" not in kickoff:
        kickoff = f"{kickoff}T00:00:00Z"

    external_key = f"openfootball:{competition_code}:{home_name}-{away_name}-{kickoff}"
    match_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, external_key))

    db.table("matches").upsert({
        "id":          match_uuid,
        "home_id":     home_id,
        "away_id":     away_id,
        "kickoff":     kickoff,
        "venue":       match.get("city") or match.get("stadium") or "",
        "stage":       stage,
        "competition": competition_code,
        "is_neutral":  True,
        "home_goals":  home_goals,
        "away_goals":  away_goals,
        "completed":   home_goals is not None,
    }, on_conflict="id").execute()
    return True


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

async def refresh_competition(competition_code: str) -> int:
    """Pull all matches for one competition from openfootball. Returns count."""
    url = SOURCES.get(competition_code)
    if not url:
        log.error("openfootball.unknown_competition", code=competition_code)
        return 0

    written = 0
    async with httpx.AsyncClient() as client:
        try:
            data = await _fetch_json(client, url)
        except Exception as e:
            log.error("openfootball.fetch_failed", code=competition_code, error=str(e))
            return 0

        # openfootball structures: {rounds: [{name, matches: [...]}]} or top-level {matches: [...]}
        rounds = data.get("rounds") or [{"name": "", "matches": data.get("matches", [])}]
        for rnd in rounds:
            round_name = rnd.get("name", "")
            stage_default = STAGE_MAP.get(round_name, "group" if "Group" in round_name else "qual")
            for m in rnd.get("matches", []):
                try:
                    if await _upsert_match(m, competition_code, stage_default):
                        written += 1
                except Exception as e:
                    log.warning("openfootball.match_failed", error=str(e))

    log.info("openfootball.competition_done", code=competition_code, written=written)
    return written


async def refresh_all() -> int:
    """Backfill every supported historical competition."""
    log.info("openfootball.refresh.start")
    total = 0
    for code in SOURCES:
        total += await refresh_competition(code)
    log.info("openfootball.refresh.done", total=total)
    return total
