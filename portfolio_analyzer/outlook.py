"""Rule-based bull/bear outlook plus optional LLM narration. No network, no anthropic import."""

from __future__ import annotations

import json
import math
from typing import Any

from portfolio_analyzer.models import CandleReading, Outlook, StopSuggestion, TickerData

RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
NEAR_EXTREME_PCT = 5.0  # "within 5%" of the 52-week low/high
HIGH_RISK_PCT = 10.0  # stop risk_pct above this means high volatility
YEAR_ROWS = 252
PE_ASSET_CLASSES = ("equity",)
LOW_ASSET_CLASSES = ("equity", "etf")  # near-52-week-low bullish rule
NEUTRAL_BULL = "No strong bullish signals"
NEUTRAL_BEAR = "No strong bearish signals"
MAX_TOKENS = 1500  # headroom for models that think before writing; prompt asks for ~120 words
SYSTEM_PROMPT = (
    "You are a cautious market commentator. Write a short plain-English outlook for the "
    "ticker using ONLY the facts supplied. Use no figures other than those supplied; do not "
    "invent prices, percentages or dates. Stay under about 120 words. End by saying this is "
    "not financial advice."
)


class NarrationError(Exception):
    """The narration API call failed."""


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _insufficient(note: str | None) -> bool:
    return bool(note) and "insufficient data" in note.lower()


def _year_range(data: TickerData) -> tuple[float, float] | None:
    """52-week (low, high): both from fundamentals, else both from >= 252 price rows."""
    f = data.fundamentals or {}
    low, high = _num(f.get("fiftyTwoWeekLow")), _num(f.get("fiftyTwoWeekHigh"))
    if low is None or high is None:
        df = data.prices
        if df is None or len(df) < YEAR_ROWS:
            return None
        tail = df.tail(YEAR_ROWS)
        lo_col = "Low" if "Low" in tail.columns else "Close"
        hi_col = "High" if "High" in tail.columns else "Close"
        if lo_col not in tail.columns or hi_col not in tail.columns:
            return None
        low, high = _num(tail[lo_col].min()), _num(tail[hi_col].max())
    if low is None or high is None or low <= 0 or high <= low:
        return None  # missing or degenerate range: proximity means nothing
    return low, high


def _last_price(data: TickerData) -> float | None:
    df = data.prices
    if df is None or "Close" not in df.columns:
        return None
    s = df["Close"].dropna()
    return _num(s.iloc[-1]) if len(s) else None


def build_outlook(
    data: TickerData,
    candles: CandleReading,
    stop: StopSuggestion,
    sector_median_pe: float | None = None,
) -> Outlook:
    bull: list[str] = []
    bear: list[str] = []

    if candles.trend == "up":
        bull.append("Uptrend: price is trending up")
    elif candles.trend == "down":
        bear.append("Downtrend: price is trending down")

    rsi = _num(candles.rsi)
    if rsi is not None:
        if rsi < RSI_OVERSOLD:
            bull.append(f"RSI {rsi:.1f} is below {RSI_OVERSOLD:.0f} (oversold)")
        elif rsi > RSI_OVERBOUGHT:
            bear.append(f"RSI {rsi:.1f} is above {RSI_OVERBOUGHT:.0f} (overbought)")

    if candles.bias == "bullish":
        bull.append("Recent candle patterns lean bullish")
    elif candles.bias == "bearish":
        bear.append("Recent candle patterns lean bearish")

    price = _last_price(data)
    rng = _year_range(data)
    if price is not None and rng is not None:
        low, high = rng
        low_gap = round((price / low - 1) * 100, 9)  # % above the low (negative: below)
        high_gap = round((1 - price / high) * 100, 9)  # % below the high (negative: above)
        near_low = low_gap <= NEAR_EXTREME_PCT and data.asset_class in LOW_ASSET_CLASSES
        near_high = high_gap <= NEAR_EXTREME_PCT
        if near_low and near_high:  # narrow range: keep only the nearer extreme
            near_low, near_high = low_gap < high_gap, high_gap < low_gap
        if near_low:
            where = f"{low_gap:.1f}% above" if low_gap > 0 else "at/below"
            bull.append(f"Price {price:.2f} is {where} its 52-week low {low:.2f}")
        if near_high:
            where = f"{high_gap:.1f}% below" if high_gap > 0 else "at/above"
            bear.append(f"Price {price:.2f} is {where} its 52-week high {high:.2f}")

    risk = _num(stop.risk_pct)
    if risk is not None and risk > HIGH_RISK_PCT:
        bear.append(f"Suggested stop risk is {risk:.1f}% of price (high volatility)")

    pe = _num((data.fundamentals or {}).get("trailingPE"))
    med = _num(sector_median_pe)
    if data.asset_class in PE_ASSET_CLASSES and pe is not None and med is not None:
        if pe < med:
            bull.append(f"P/E {pe:.1f} is below the sector median {med:.1f}")
        elif pe > med:
            bear.append(f"P/E {pe:.1f} is above the sector median {med:.1f}")

    notes = []
    if _insufficient(candles.note):
        notes.append(f"Technical indicators: {candles.note}")
    if _insufficient(stop.note):
        notes.append(f"Stop analysis: {stop.note}")
    if not bull:
        bull = notes or [NEUTRAL_BULL]
    if not bear:
        bear = notes or [NEUTRAL_BEAR]
    return Outlook(data.ticker, bull, bear, None)


def _text(content: Any) -> str:
    if not isinstance(content, (list, tuple)):
        return ""
    return "".join(t for t in (getattr(b, "text", None) for b in content) if isinstance(t, str))


def narrate(
    outlook: Outlook, facts: dict[str, Any], client: Any, model: str = "claude-sonnet-5-5"
) -> str:
    """Ask the model to narrate `facts` (only) in plain English; `client` is injected."""
    prompt = "Facts (JSON):\n" + json.dumps(facts, default=str, indent=2)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        raise NarrationError(f"narration failed for {outlook.ticker}: {exc}") from exc
    stop = getattr(resp, "stop_reason", None)
    if stop in ("max_tokens", "refusal"):
        raise NarrationError(f"narration for {outlook.ticker} stopped early: stop_reason={stop}")
    text = _text(getattr(resp, "content", None)).strip()
    if not text:
        raise NarrationError(f"narration for {outlook.ticker} returned no text")
    return text
