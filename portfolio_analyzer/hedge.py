"""Portfolio hedge analysis. Pure functions over DataFrames; no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_analyzer.models import HedgeResult, Holding, StressResult, TickerData

HIGH_CORR = 0.85
OFFSET_CORR = -0.2
OFFSET_BETA = 0.3
CONCENTRATION_LIMIT = 0.5
EFFECTIVE_BETA = 0.7
PARTIAL_BETA = 1.0
MIN_OBS = 2
WINDOW_TOLERANCE = pd.Timedelta(days=5)  # first/last trading day may sit just inside a window
STRESS_WINDOWS = {
    "covid_2020": ("2020-02-19", "2020-03-23"),
    "bear_2022": ("2022-01-03", "2022-10-12"),
}
_BENCH = "__benchmark__"


def _close_returns(td: TickerData) -> pd.Series:
    close = td.prices["Close"].astype(float)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.pct_change()


def joined_returns(
    holdings: list[Holding], data: dict[str, TickerData], benchmark: TickerData
) -> pd.DataFrame:
    """Daily Close returns per holding plus benchmark, inner-joined on dates, NaN-free."""
    cols = {h.ticker: _close_returns(data[h.ticker]) for h in holdings if h.ticker in data}
    cols[_BENCH] = _close_returns(benchmark)
    joined = pd.concat(cols, axis=1, join="inner")
    return joined.replace([np.inf, -np.inf], np.nan).dropna()


def _beta(series: pd.Series, bench: pd.Series) -> float:
    var = float(bench.var())
    if not np.isfinite(var) or var <= 1e-18:
        return 0.0
    return float(series.cov(bench) / var)


def _concentration(holdings: list[Holding], data: dict[str, TickerData]) -> dict[str, float]:
    out: dict[str, float] = {}
    for h in holdings:
        td = data[h.ticker]
        if td.asset_class == "equity":
            key = str((td.fundamentals or {}).get("sector") or "Unknown")
        else:
            key = td.asset_class
        out[key] = out.get(key, 0.0) + h.weight
    return out


def _stress(port: pd.Series, bench: pd.Series) -> list[StressResult]:
    results = []
    for name, (start, end) in STRESS_WINDOWS.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if len(port) == 0 or port.index.min() > s + WINDOW_TOLERANCE or (
            port.index.max() < e - WINDOW_TOLERANCE
        ):
            results.append(StressResult(name, None, None, "joined history does not cover window"))
            continue
        p, b = port.loc[s:e], bench.loc[s:e]
        if len(p) == 0:
            results.append(StressResult(name, None, None, "no overlapping data in window"))
            continue
        results.append(
            StressResult(name, float((1 + p).prod() - 1), float((1 + b).prod() - 1), None)
        )
    return results


def _offsets(rets: pd.DataFrame, weights: pd.Series, bench: pd.Series) -> list[str]:
    out = []
    for t in weights.index:
        others = weights.drop(t)
        rho = np.nan
        if len(others) and others.sum() > 0:
            rest = (rets[others.index] * (others / others.sum())).sum(axis=1)
            rho = rets[t].corr(rest)
        if (np.isfinite(rho) and rho <= OFFSET_CORR) or _beta(rets[t], bench) < OFFSET_BETA:
            out.append(t)
    return out


def _safe(tickers: list[str], reason: str, conc: dict[str, float]) -> HedgeResult:
    corr = pd.DataFrame(np.nan, index=tickers, columns=tickers, dtype=float)
    return HedgeResult(0.0, corr, [], conc, [], [], "none", reason)


def analyse_hedge(
    holdings: list[Holding], data: dict[str, TickerData], benchmark: TickerData
) -> HedgeResult:
    held = [h for h in holdings if h.ticker in data]
    tickers = [h.ticker for h in held]
    conc = _concentration(held, data) if held else {}
    rets = joined_returns(held, data, benchmark) if held else pd.DataFrame()
    if not held or len(rets) < MIN_OBS:
        return _safe(tickers, "insufficient overlapping history to assess hedge", conc)

    weights = pd.Series({h.ticker: h.weight for h in held})
    bench = rets[_BENCH]
    port = (rets[tickers] * weights).sum(axis=1)
    beta = _beta(port, bench)
    corr = rets[tickers].corr()
    pairs = [
        (a, b, float(corr.loc[a, b]))
        for i, a in enumerate(tickers)
        for b in tickers[i + 1 :]
        if np.isfinite(corr.loc[a, b]) and corr.loc[a, b] >= HIGH_CORR
    ]
    stress = _stress(port, bench)
    avail = [s for s in stress if s.skipped_reason is None]
    wins = [s for s in avail if s.portfolio_return > s.benchmark_return]

    if beta <= EFFECTIVE_BETA and avail and len(wins) == len(avail):
        verdict = "effective"
    elif beta < PARTIAL_BETA or wins:
        verdict = "partial"
    else:
        verdict = "none"

    stress_txt = (
        f"beats the benchmark in {len(wins)} of {len(avail)} stress windows"
        if avail
        else "no stress window is covered by the data"
    )
    reason = f"Portfolio beta of {beta:.2f} vs the benchmark and it {stress_txt}"
    flagged = [f"{k} {v:.0%}" for k, v in conc.items() if v > CONCENTRATION_LIMIT]
    if flagged:
        reason += f"; concentration over 50% in {', '.join(flagged)}"
    reason += "."
    return HedgeResult(
        beta, corr, pairs, conc, _offsets(rets, weights, bench), stress, verdict, reason
    )
