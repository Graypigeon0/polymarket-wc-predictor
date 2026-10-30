"""Candlestick pattern detection. Pure functions over DataFrames; no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_analyzer.models import CandleReading, Direction, PatternHit, TickerData

MIN_BARS = 60
LOOKBACK = 3  # patterns on the last N bars drive the bias
_OHLC = ("Open", "High", "Low", "Close")


def rsi(close: pd.Series, period: int = 14) -> float | None:
    """Wilder RSI of the last bar. None if too short or no movement at all."""
    close = close.dropna()
    if len(close) <= period:
        return None
    delta = close.diff().dropna()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if gain == 0 and loss == 0:
        return None  # zero volatility: RSI undefined
    if loss == 0:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + gain / loss))


def _trend(close: pd.Series) -> str:
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    last = close.iloc[-1]
    if np.isnan(sma20) or np.isnan(sma50):
        return "sideways"
    if last > sma20 > sma50:
        return "up"
    if last < sma20 < sma50:
        return "down"
    return "sideways"


def _neutral(ticker: str, note: str, trend: str = "sideways", r: float | None = None):
    return CandleReading(ticker, [], "neutral", trend, r, note)  # type: ignore[arg-type]


def _pattern_flags(df: pd.DataFrame) -> dict[str, tuple[Direction, pd.Series]]:
    o, h, lo, c = (df[k].astype(float) for k in _OHLC)
    body = (c - o).abs()
    rng = h - lo
    top = pd.concat([o, c], axis=1).max(axis=1)
    bot = pd.concat([o, c], axis=1).min(axis=1)
    upper = h - top
    lower = bot - lo
    bull = c > o
    bear = c < o
    valid = rng > 0
    sma20 = c.rolling(20).mean()
    # trend context is what the market was doing going into the bar
    falling = sma20.shift(1) < sma20.shift(2)
    rising = sma20.shift(1) > sma20.shift(2)

    doji = valid & (body <= 0.1 * rng)
    hammer = valid & (lower >= 2 * body) & (upper <= body) & (lower > 0) & falling
    star = valid & (upper >= 2 * body) & (lower <= body) & (upper > 0) & rising

    po, pc, pbody = o.shift(1), c.shift(1), body.shift(1)
    bull_eng = (
        bull & bear.shift(1, fill_value=False) & (pbody > 0) & (o <= pc) & (c >= po) & (body > pbody)
    )
    bear_eng = (
        bear & bull.shift(1, fill_value=False) & (pbody > 0) & (o >= pc) & (c <= po) & (body > pbody)
    )

    # three-bar reversals: bar1 = shift(2), bar2 = shift(1), bar3 = current
    b1_body = body.shift(2)
    b1_mid = (o.shift(2) + c.shift(2)) / 2
    small_mid = (b1_body > 0) & (body.shift(1) <= 0.3 * b1_body)
    morning = bear.shift(2, fill_value=False) & small_mid & bull & (c > b1_mid)
    evening = bull.shift(2, fill_value=False) & small_mid & bear & (c < b1_mid)

    bull1, bull2 = bull.shift(1, fill_value=False), bull.shift(2, fill_value=False)
    bear1, bear2 = bear.shift(1, fill_value=False), bear.shift(2, fill_value=False)
    soldiers = bull & bull1 & bull2 & (c > pc) & (pc > c.shift(2))
    crows = bear & bear1 & bear2 & (c < pc) & (pc < c.shift(2))

    return {
        "doji": ("neutral", doji),
        "hammer": ("bullish", hammer),
        "shooting star": ("bearish", star),
        "bullish engulfing": ("bullish", bull_eng),
        "bearish engulfing": ("bearish", bear_eng),
        "morning star": ("bullish", morning),
        "evening star": ("bearish", evening),
        "three white soldiers": ("bullish", soldiers),
        "three black crows": ("bearish", crows),
    }


def analyse_candles(data: TickerData) -> CandleReading:
    df = data.prices
    if df is None or len(df) < MIN_BARS or any(k not in df.columns for k in _OHLC):
        return _neutral(data.ticker, "insufficient data: need at least 60 bars of history")
    df = df.dropna(subset=list(_OHLC))
    if len(df) < MIN_BARS:
        return _neutral(data.ticker, "insufficient data: need at least 60 bars of history")

    trend = _trend(df["Close"].astype(float))
    r = rsi(df["Close"].astype(float))

    hits: list[PatternHit] = []
    n = len(df)
    for name, (direction, flags) in _pattern_flags(df).items():
        recent = flags.to_numpy()[n - LOOKBACK :]
        for off, hit in enumerate(recent):
            if hit:
                hits.append(PatternHit(name, direction, df.index[n - LOOKBACK + off]))
    hits.sort(key=lambda p: p.date)

    bulls = sum(p.direction == "bullish" for p in hits)
    bears = sum(p.direction == "bearish" for p in hits)
    bias: Direction = "bullish" if bulls > bears else "bearish" if bears > bulls else "neutral"
    return CandleReading(data.ticker, hits, bias, trend, r, None)  # type: ignore[arg-type]
