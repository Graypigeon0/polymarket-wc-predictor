"""
openfootball ingestion — free historical match data from GitHub.

openfootball is a crowdsourced repo of football data in clean JSON format,
no API key, no rate limit, no paywall. Used for historical backtest data.

Source: https://github.com/openfootball/worldcup.json
Format (confirmed Nov 2026):
  {
    "name": "World Cup 2022",
    "matches": [
      {
        "round": "Matchday 1" | "Round of 16" | "Final" | ...
        "date": "2022-11-20",
        "time": "19:00",
        "team1": "Qatar",
        "team2": "Ecuador",
        "group": "Group A",          # only for group stage
        "ground": "Al Bayt Stadium, Al Khor",
        "goals1": [...],             # array of goal events
        "goals2": [...],
        "score": {...}               # may be empty if not finished
      },
      ...
    ]
  }

Score is derived from len(goals1) / len(goals2).
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.db.client import get_client
from backend.ingestion.football_data import _confederation, _fifa_code

log = structlog.get_logger()

SOURCES: dict[str, str] = {
    "WC2022":   "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json",
    "WC2018":   "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2018/worldcup.json",
    "WC2014":   "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2014/worldcup.json",
    "EURO2024": "https://raw.githubusercontent.com/openfootball/euro.json/master/2024/euro.json",
    "EURO2020": "https://raw.githubusercontent.com/openfootball/euro.json/master/2020/euro.json",
}

# Round names → internal stage codes
ROUND_TO_STAGE = {
    "round of 16":            "r16",
    "round of sixteen":       "r16",
    "quarter-finals":         "qf",
    "quarterfinals":          "qf",
    "quarter finals":         "qf",
    "semi-finals":            "sf",
    "semifinals":             "sf",
    "semi finals":            "sf",
    "third-place play-off":   "3p",
    "third place play-off":   "3p",
    "match for third place":  "3p",
    "third place":            "3p",
    "final":                  "final",
}


def _stage_from_round(round_name: str, group: str | None) -> str:
    """Map an openfootball round/group label to our internal stage code."""
    rn = (round_name or "").strip().lower()

    # Direct knockout match
    if rn in ROUND_TO_STAGE:
        return ROUND_TO_STAGE[rn]

    # Group stage: "Matchday N" + group field, or "Group A" round name
    if group or rn.startswith("matchday") or rn.startswith("group"):
        return "group"

    # Knockout patterns that contain extra words
    for key, stage in ROUND_TO_STAGE.items():
        if key in rn:
            return stage

    return "group"  # safe default


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
async def _fetch_json(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    r = await client.get(url, timeout=30.0)
    r.raise_for_status()
    return r.json()


async def _upsert_team(country: str) -> str:
    """Get or create a team by country name. Returns the team UUID."""
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


def _score_from_match(match: dict) -> tuple[int | None, int | None]:
    """Derive (home_goals, away_goals) from openfootball match payload."""
    # Preferred: explicit score dict
    score = match.get("score")
    if isinstance(score, dict):
        ft = score.get("ft") or score.get("fullTime")
        if isinstance(ft, list) and len(ft) == 2:
            try:
                return int(ft[0]), int(ft[1])
            except (TypeError, ValueError):
                pass

    # Fallback: count goal events
    goals1 = match.get("goals1")
    goals2 = match.get("goals2")
    if isinstance(goals1, list) and isinstance(goals2, list):
        return len(goals1), len(goals2)

    return None, None


async def _upsert_match(match: dict, competition_code: str) -> bool:
    """Upsert one openfootball match. Returns True on success, False otherwise."""
    db = get_client()

    home_name = match.get("team1")
    away_name = match.get("team2")
    date_str = match.get("date")
    if not home_name or not away_name or not date_str:
        return False

    # Build a valid ISO timestamp
    time_str = match.get("time") or "00:00"
    # openfootball times look like "19:00"; sanitize to HH:MM
    m = re.match(r"(\d{1,2}:\d{2})", str(time_str))
    time_clean = m.group(1) if m else "00:00"
    # Pad to HH:MM if needed
    h, mm = time_clean.split(":")
    kickoff = f"{date_str}T{h.zfill(2)}:{mm}:00Z"

    home_id = await _upsert_team(str(home_name))
    away_id = await _upsert_team(str(away_name))
    home_goals, away_goals = _score_from_match(match)
    stage = _stage_from_round(match.get("round", ""), match.get("group"))

    external_key = f"openfootball:{competition_code}:{home_name}-{away_name}-{date_str}"
    match_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, external_key))

    row = {
        "id":          match_uuid,
        "home_id":     home_id,
        "away_id":     away_id,
        "kickoff":     kickoff,
        "venue":       match.get("ground") or "",
        "stage":       stage,
        "competition": competition_code,
        "is_neutral":  True,
        "home_goals":  home_goals,
        "away_goals":  away_goals,
        "completed":   home_goals is not None,
    }
    db.table("matches").upsert(row, on_conflict="id").execute()
    return True


async def refresh_competition(competition_code: str) -> int:
    """Pull all matches for one competition from openfootball."""
    url = SOURCES.get(competition_code)
    if not url:
        log.error("openfootball.unknown_competition", code=competition_code)
        return 0

    written = 0
    failed = 0
    async with httpx.AsyncClient() as client:
        try:
            data = await _fetch_json(client, url)
        except Exception as e:
            log.error("openfootball.fetch_failed", code=competition_code, error=str(e))
            return 0

        matches = data.get("matches", [])
        log.info("openfootball.fetched", code=competition_code, total=len(matches))

        for m in matches:
            try:
                if await _upsert_match(m, competition_code):
                    written += 1
            except Exception as e:
                failed += 1
                # Only log first 3 failures to avoid spamming
                if failed <= 3:
                    log.warning("openfootball.match_failed",
                                code=competition_code,
                                team1=m.get("team1"), team2=m.get("team2"),
                                error=str(e)[:300])

    log.info("openfootball.competition_done",
             code=competition_code, written=written, failed=failed)
    return written


async def refresh_all() -> int:
    """Backfill every supported historical competition."""
    log.info("openfootball.refresh.start")
    total = 0
    for code in SOURCES:
        total += await refresh_competition(code)
    log.info("openfootball.refresh.done", total=total)
    return total
