"""ATR and swing-low support stop losses. Pure functions over DataFrames; no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_analyzer.models import StopSuggestion, TickerData

MIN_BARS = 60
ATR_PERIOD = 14
ATR_MULTIPLE = 2.0
SWING_WINDOW = 5  # bars on each side that must not undercut a swing-low
SWING_LOOKBACK = 60  # only swing-lows within the last N bars count
SUPPORT_BUFFER = 0.99  # stop sits 1% under the swing-low
_COLS = ("High", "Low", "Close")


def atr(prices: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder average true range, aligned to the input index."""
    high, low, close = (prices[k].astype(float) for k in _COLS)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _swing_low(low: pd.Series, price: float) -> float | None:
    """Nearest (highest) swing-low below `price` among the last SWING_LOOKBACK bars.

    A swing-low is a bar whose Low is the lowest of the SWING_WINDOW bars either side and
    strictly below at least one of them (a flat plateau is not a swing).
    """
    vals = low.to_numpy(dtype=float)
    n = len(vals)
    w = SWING_WINDOW
    best: float | None = None
    for i in range(max(w, n - SWING_LOOKBACK), n - w):
        win = vals[i - w : i + w + 1]
        if vals[i] == win.min() and win.max() > vals[i] and vals[i] < price:
            if best is None or vals[i] > best:
                best = float(vals[i])
    return best


def _na(ticker: str, note: str) -> StopSuggestion:
    return StopSuggestion(ticker, None, None, None, None, "n/a", note)


def suggest_stop(data: TickerData) -> StopSuggestion:
    df = data.prices
    if df is None or any(k not in df.columns for k in _COLS):
        return _na(data.ticker, "insufficient data: need High, Low and Close history")
    df = df.dropna(subset=list(_COLS))
    if len(df) < MIN_BARS:
        return _na(data.ticker, "insufficient data: need at least 60 bars of history")

    price = float(df["Close"].iloc[-1])
    a = float(atr(df).iloc[-1])
    if not np.isfinite(a) or not np.isfinite(price) or price <= 0:
        return _na(data.ticker, "insufficient data: price or ATR is not usable")

    atr_stop = price - ATR_MULTIPLE * a
    low = _swing_low(df["Low"], price)
    support_stop = low * SUPPORT_BUFFER if low is not None else None

    notes: list[str] = []
    if a == 0:
        notes.append("zero volatility: ATR is 0, stop equals price")
    if support_stop is None:
        notes.append("no swing-low below price in last 60 bars; ATR stop only")

    # tighter (higher) stop, but only if it leaves more than 1 ATR of room
    if support_stop is not None and support_stop > atr_stop and price - support_stop > a:
        recommended, method = support_stop, "support"
    else:
        recommended, method = atr_stop, "atr"

    risk_pct = (price - recommended) / price * 100.0
    return StopSuggestion(
        data.ticker,
        atr_stop,
        support_stop,
        recommended,
        risk_pct,
        method,  # type: ignore[arg-type]
        "; ".join(notes) or None,
    )
