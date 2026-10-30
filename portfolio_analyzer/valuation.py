"""Underpriced equity screen. Pure over TickerData; only load_universe("sp500") hits the network."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_analyzer.models import TickerData, ValuationCandidate

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "Mozilla/5.0 (compatible; portfolio-analyzer/1.0)"
SMA_WINDOW = 200
TREND_FLOOR = 0.9  # Close must be >= 0.9 x SMA200
YEAR_ROWS = 252
UNKNOWN_SECTOR = "Unknown"
NEUTRAL_RANK = 0.5  # rank for a factor with a single peer (avoids 0/0 -> NaN)
SCORE_DECIMALS = 4


def load_universe(spec: str) -> list[str]:
    """ "sp500" scrapes Wikipedia; anything else is a file of tickers (one per line, # comments)."""
    if spec.strip().lower() == "sp500":
        tables = pd.read_html(SP500_URL, storage_options={"User-Agent": USER_AGENT})
        symbols = tables[0]["Symbol"].astype(str)
        return [s.strip().upper().replace(".", "-") for s in symbols if s.strip()]
    out: list[str] = []
    for line in Path(spec).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.upper())
    return out


def _num(value: Any) -> float | None:
    """Finite float or None (missing, non-numeric, NaN, inf)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def has_fundamentals(data: TickerData) -> bool:
    """True only for an equity with trailingPE, freeCashflow and marketCap all present."""
    if data.asset_class != "equity":
        return False
    return all(
        _num(data.fundamentals.get(k)) is not None
        for k in ("trailingPE", "freeCashflow", "marketCap")
    )


def _last_close(data: TickerData) -> float | None:
    if "Close" not in data.prices:
        return None
    close = data.prices["Close"].astype(float).dropna()
    return float(close.iloc[-1]) if len(close) else None


def _trend_ok(data: TickerData) -> bool:
    # Fewer than SMA_WINDOW rows: SMA200 is undefined, so the ticker fails the filter (excluded).
    if "Close" not in data.prices:
        return False
    close = data.prices["Close"].astype(float).dropna()
    if len(close) < SMA_WINDOW:
        return False
    return bool(close.iloc[-1] >= TREND_FLOOR * close.iloc[-SMA_WINDOW:].mean())


def _range_position(data: TickerData) -> float | None:
    """Position in the 52-week range (0 = at low). Falls back to the last 252 price rows."""
    close = _last_close(data)
    hi = _num(data.fundamentals.get("fiftyTwoWeekHigh"))
    lo = _num(data.fundamentals.get("fiftyTwoWeekLow"))
    if (hi is None or lo is None) and {"High", "Low"} <= set(data.prices.columns):
        window = data.prices.tail(YEAR_ROWS)
        if len(window):
            hi, lo = _num(window["High"].max()), _num(window["Low"].min())
    if close is None or hi is None or lo is None or hi <= lo:
        return None
    return min(max((close - lo) / (hi - lo), 0.0), 1.0)


def _factors(data: TickerData) -> dict[str, float | None] | None:
    """Raw factor values for a qualifying equity, else None."""
    if data.asset_class != "equity":
        return None
    f = data.fundamentals
    pe, fcf = _num(f.get("trailingPE")), _num(f.get("freeCashflow"))
    if pe is None or pe <= 0 or fcf is None or fcf <= 0 or not _trend_ok(data):
        return None
    cap = _num(f.get("marketCap"))
    peg = _num(f.get("pegRatio"))
    return {
        "pe": pe,
        "peg": peg if peg is not None and peg > 0 else None,
        "fcf_yield": fcf / cap if cap is not None and cap > 0 else None,
        "range": _range_position(data),
    }


def _pct_rank(values: pd.Series, higher_is_cheaper: bool) -> pd.Series:
    """Percentile rank 0-1 among non-NaN values; higher = cheaper. Single peer -> 0.5."""
    valid = values.dropna()
    if len(valid) < 2:
        return pd.Series(NEUTRAL_RANK, index=valid.index)
    ranks = (valid.rank(method="average") - 1) / (len(valid) - 1)
    return ranks if higher_is_cheaper else 1 - ranks


def screen_underpriced(universe: dict[str, TickerData], top: int = 10) -> list[ValuationCandidate]:
    rows: dict[str, dict[str, float | None]] = {}
    sectors: dict[str, str] = {}
    for ticker, data in universe.items():
        factors = _factors(data)
        if factors is None:
            continue
        rows[ticker] = factors
        sector = data.fundamentals.get("sector")
        sectors[ticker] = str(sector) if sector else UNKNOWN_SECTOR
    if not rows:
        return []

    frame = pd.DataFrame.from_dict(rows, orient="index").astype(float)
    frame["sector"] = pd.Series(sectors)
    # (column, higher raw value is cheaper)
    directions = {"pe": False, "peg": False, "fcf_yield": True, "range": False}
    candidates: list[ValuationCandidate] = []
    for _, group in frame.groupby("sector"):
        ranks = pd.DataFrame(
            {col: _pct_rank(group[col], up) for col, up in directions.items()},
            index=group.index,
        )
        medians = group[["pe", "peg", "fcf_yield"]].median()
        for ticker in group.index:
            avail = ranks.loc[ticker].dropna()
            if avail.empty:
                continue
            row = group.loc[ticker]
            reasons = [f"P/E {row['pe']:.1f} vs sector median {medians['pe']:.1f}"]
            if not np.isnan(row["peg"]):
                reasons.append(f"PEG {row['peg']:.2f} vs sector median {medians['peg']:.2f}")
            if not np.isnan(row["fcf_yield"]):
                reasons.append(
                    f"FCF yield {row['fcf_yield']:.1%} vs sector median {medians['fcf_yield']:.1%}"
                )
            if not np.isnan(row["range"]):
                reasons.append(f"{row['range']:.0%} of the way up its 52-week range")
            score = round(float(avail.mean()), SCORE_DECIMALS)
            candidates.append(ValuationCandidate(ticker, score, reasons))
    candidates.sort(key=lambda c: (-c.score, c.ticker))
    return candidates[: max(top, 0)]
