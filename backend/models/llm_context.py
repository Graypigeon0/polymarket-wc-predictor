"""
LLM context layer.

Two-stage pipeline using Mistral AI (primary):

  Stage A: relevance classifier
    Input:  raw news item (headline + body)
    Output: {affects: bool, teams: [...], players: [...], category, severity, summary}

  Stage B: rating adjuster (only for items where affects=True)
    Input:  news item + current squad state + base ratings
    Output: {attack_delta, defense_delta, confidence, expires_at, reasoning}

Why Mistral:
  - EU-based company, works across all EU countries including Cyprus
  - Free tier available at console.mistral.ai
  - OpenAI-compatible API (easy to swap models later)
  - mistral-small-latest: fast, cheap, great at structured JSON output

Guardrails:
  - Per-event delta capped at +/-5% of base rating (config.max_event_delta_pct)
  - Big moves (>3%) require 2 independent sources within 24h
  - All deltas decay back to zero by their expires_at timestamp
  - Reasoning string stored verbatim for audit trail
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings

log = structlog.get_logger()

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


# ---------------------------------------------------------------------
# Stage A prompt
# ---------------------------------------------------------------------

CLASSIFIER_SYSTEM = """You are a football news triage classifier for a World Cup 2026 prediction model.

Given a news/social-media item, decide whether it materially affects any national team's
match prediction in the next 6 weeks.

You MUST respond with valid JSON only, no prose, matching this schema:
{
  "affects": boolean,
  "teams": [<FIFA team codes, e.g. "BRA", "FRA">],
  "players": [<player full names>],
  "category": "injury" | "lineup" | "tactical" | "morale" | "suspension" | "other",
  "severity": <integer 0-10, 0=trivial, 10=team-defining>,
  "summary": <one-sentence factual summary, no speculation>
}

Examples of affects=true: confirmed injury to first-choice player, suspension, manager change,
locked-in starting XI ahead of kickoff, public falling-out within squad.

Examples of affects=false: transfer rumour about club football, opinion pieces, historical
retrospectives, fan speculation, match reports of unrelated club games.

If unsure, return affects=false. Precision over recall."""


# ---------------------------------------------------------------------
# Stage B prompt
# ---------------------------------------------------------------------

ADJUSTER_SYSTEM = """You are a football analyst adjusting a national team's attack and defense
ratings based on a single news event.

You are given:
  - The news event (headline + body + category + severity from triage)
  - The team's current 26-man squad with positions and starter probabilities
  - The team's current base attack/defense ratings

Output a JSON object only, no prose:
{
  "attack_delta": <float, fraction of base rating, e.g. -0.03 for -3%>,
  "defense_delta": <float, same units>,
  "confidence": <0.0 to 1.0>,
  "expires_in_hours": <integer, when this delta should decay back to zero>,
  "reasoning": <2-3 sentence justification>
}

Rules:
  - Hard cap: |delta| must not exceed 0.05 (5%) per event
  - Be conservative. Most news has small impact.
  - Defense rating: lower is BETTER. Losing a star defender -> defense_delta POSITIVE
    (worse defense, higher concede rate).
  - Attack rating: higher is better. Losing a striker -> attack_delta NEGATIVE.
  - expires_in_hours: short injury -> next match only; long-term -> tournament duration.
"""


@dataclass
class ClassifierOutput:
    affects: bool
    teams: list[str]
    players: list[str]
    category: str
    severity: int
    summary: str


@dataclass
class AdjusterOutput:
    attack_delta: float
    defense_delta: float
    confidence: float
    expires_at: datetime
    reasoning: str


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {get_settings().mistral_api_key}",
        "Content-Type": "application/json",
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
async def _chat_json(system: str, user: str, temperature: float = 0.0) -> dict:
    """Single Mistral chat completion returning parsed JSON."""
    s = get_settings()
    payload = {
        "model": s.mistral_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{MISTRAL_BASE_URL}/chat/completions",
            headers=_headers(),
            json=payload,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"] or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            log.warning("llm.json_parse_failed", content=content[:200])
            return {}


async def classify(headline: str, body: str) -> ClassifierOutput:
    """Stage A: relevance classifier."""
    user = f"HEADLINE: {headline}\n\nBODY: {body}"
    data = await _chat_json(CLASSIFIER_SYSTEM, user, temperature=0.0)
    return ClassifierOutput(
        affects=bool(data.get("affects", False)),
        teams=data.get("teams", []),
        players=data.get("players", []),
        category=data.get("category", "other"),
        severity=int(data.get("severity", 0)),
        summary=data.get("summary", ""),
    )


async def adjust(
    headline: str,
    body: str,
    team_code: str,
    squad_summary: str,
    base_attack: float,
    base_defense: float,
) -> AdjusterOutput:
    """Stage B: rating adjuster."""
    s = get_settings()
    user = f"""TEAM: {team_code}
BASE ATTACK RATING: {base_attack:.4f}
BASE DEFENSE RATING: {base_defense:.4f}

SQUAD:
{squad_summary}

NEWS:
{headline}

{body}
"""
    data = await _chat_json(ADJUSTER_SYSTEM, user, temperature=0.2)

    cap = s.max_event_delta_pct
    a_delta = max(-cap, min(cap, float(data.get("attack_delta", 0.0))))
    d_delta = max(-cap, min(cap, float(data.get("defense_delta", 0.0))))

    hours = max(1, int(data.get("expires_in_hours", 72)))
    return AdjusterOutput(
        attack_delta=a_delta,
        defense_delta=d_delta,
        confidence=float(data.get("confidence", 0.5)),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=hours),
        reasoning=data.get("reasoning", ""),
    )


async def classify_and_adjust(items: list) -> None:
    """
    Full pipeline: classify each news item, then adjust ratings for affected teams.
    Writes news_events + rating_deltas rows.
    """
    log.info("llm_context.pipeline.todo", item_count=len(items))
    # TODO:
    #   for each item:
    #     1. Skip if URL already in news_events (dedupe)
    #     2. Stage A classify
    #     3. Store news_events row
    #     4. If affects: for each team in classifier output, run Stage B,
    #        store rating_deltas row
    #     5. Mark older overlapping deltas as superseded
