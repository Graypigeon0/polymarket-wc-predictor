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
# A window counts as covered when the joined data starts no later than 5 days after the window
# start and ends no earlier than 5 days before its end (holidays/weekends). Within that
# tolerance the stress return is computed on the available slice.
WINDOW_TOLERANCE = pd.Timedelta(days=5)
BETA_DECIMALS = 6  # beta is rounded before threshold checks so clones are exactly 1.0
MIN_VARIANCE = 1e-18
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
    return float(series.cov(bench) / bench.var())


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


def _skip_reason(
    port: pd.Series, starts: dict[str, pd.Timestamp], window_start: pd.Timestamp
) -> str:
    latest = max(starts, key=lambda t: starts[t])
    if starts[latest] > window_start:
        return f"history for {latest} starts {starts[latest].date()}, after the window start"
    return f"joined history ({port.index.min().date()} to {port.index.max().date()}) "\
        "does not cover the window"


def _stress(
    port: pd.Series, bench: pd.Series, starts: dict[str, pd.Timestamp]
) -> list[StressResult]:
    results = []
    for name, (start, end) in STRESS_WINDOWS.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if port.index.min() > s + WINDOW_TOLERANCE or port.index.max() < e - WINDOW_TOLERANCE:
            results.append(StressResult(name, None, None, _skip_reason(port, starts, s)))
            continue
        # returns strictly after the start date: close(start) -> close(end)
        mask = (port.index > s) & (port.index <= e)
        if not mask.any():
            results.append(StressResult(name, None, None, "no overlapping data in window"))
            continue
        results.append(
            StressResult(
                name, float((1 + port[mask]).prod() - 1), float((1 + bench[mask]).prod() - 1), None
            )
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
    missing = [h.ticker for h in holdings if h.ticker not in data]
    held = [h for h in holdings if h.ticker in data]
    note = f"excluded from analysis (no data): {', '.join(missing)}" if missing else ""
    total = sum(h.weight for h in held)
    if not held or total <= 0:
        reason = f"No holdings with data to analyse; {note}." if note else "No holdings to analyse."
        return _safe([], reason, {})
    held = [Holding(h.ticker, h.weight / total) for h in held]  # renormalise to 1.0
    tickers = [h.ticker for h in held]
    conc = _concentration(held, data)

    def with_note(text: str) -> str:
        return f"{text}; {note}." if note else f"{text}."

    rets = joined_returns(held, data, benchmark)
    if len(rets) < MIN_OBS:
        return _safe(tickers, with_note("Insufficient overlapping history to assess hedge"), conc)
    bench = rets[_BENCH]
    var = float(bench.var())
    if not np.isfinite(var) or var <= MIN_VARIANCE:
        return _safe(
            tickers, with_note("Benchmark has no variance over the overlapping history"), conc
        )

    weights = pd.Series({h.ticker: h.weight for h in held})
    port = (rets[tickers] * weights).sum(axis=1)
    beta = round(_beta(port, bench), BETA_DECIMALS)
    corr = rets[tickers].corr()
    pairs = [
        (a, b, float(corr.loc[a, b]))
        for i, a in enumerate(tickers)
        for b in tickers[i + 1 :]
        if np.isfinite(corr.loc[a, b]) and corr.loc[a, b] >= HIGH_CORR
    ]
    starts = {t: data[t].prices.index.min() for t in tickers}
    starts[_BENCH] = benchmark.prices.index.min()
    starts = {("benchmark" if k == _BENCH else k): v for k, v in starts.items()}
    stress = _stress(port, bench, starts)
    avail = [s for s in stress if s.skipped_reason is None]
    wins = [s for s in avail if s.portfolio_return > s.benchmark_return]

    if beta <= EFFECTIVE_BETA and avail and len(wins) == len(avail):
        verdict = "effective"
    elif beta < PARTIAL_BETA or wins:
        verdict = "partial"
    else:
        verdict = "none"

    if avail:
        stress_txt = f"it beat the benchmark in {len(wins)} of {len(avail)} tested stress windows"
    else:
        stress_txt = "no stress window could be tested"
        if beta <= EFFECTIVE_BETA:
            stress_txt += ", so an effective verdict was withheld"
    text = f"Portfolio beta is {beta:.2f} and {stress_txt}"
    flagged = [f"{k} {v:.0%}" for k, v in conc.items() if v > CONCENTRATION_LIMIT]
    if flagged:
        text += f"; concentration over 50% in {', '.join(flagged)}"
    if note:
        text += f"; {note}"
    return HedgeResult(
        beta, corr, pairs, conc, _offsets(rets, weights, bench), stress, verdict, text + "."
    )
