from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

AssetClass = Literal["equity", "etf", "commodity", "crypto"]
Direction = Literal["bullish", "bearish", "neutral"]


@dataclass
class Holding:
    ticker: str
    weight: float


@dataclass
class TickerData:
    ticker: str
    asset_class: AssetClass
    prices: pd.DataFrame  # columns Open, High, Low, Close, Volume; DatetimeIndex
    fundamentals: dict[str, Any]


@dataclass
class PatternHit:
    name: str
    direction: Direction
    date: pd.Timestamp


@dataclass
class CandleReading:
    ticker: str
    patterns: list[PatternHit]
    bias: Direction
    trend: Literal["up", "down", "sideways"]
    rsi: float | None
    note: str | None


@dataclass
class StopSuggestion:
    ticker: str
    atr_stop: float | None
    support_stop: float | None
    recommended: float | None
    risk_pct: float | None
    method: Literal["atr", "support", "n/a"]
    note: str | None


@dataclass
class StressResult:
    name: str
    portfolio_return: float | None
    benchmark_return: float | None
    skipped_reason: str | None


@dataclass
class HedgeResult:
    beta: float
    correlation: pd.DataFrame
    high_corr_pairs: list[tuple[str, str, float]]
    concentration: dict[str, float]
    offsets: list[str]
    stress: list[StressResult]
    verdict: Literal["effective", "partial", "none"]
    reason: str


@dataclass
class ValuationCandidate:
    ticker: str
    score: float
    reasons: list[str]


@dataclass
class Outlook:
    ticker: str
    bullish: list[str]
    bearish: list[str]
    narrative: str | None
