from __future__ import annotations

import math
from dataclasses import dataclass

from portfolio_analyzer.models import (
    CandleReading,
    HedgeResult,
    Holding,
    Outlook,
    StopSuggestion,
    TickerData,
    ValuationCandidate,
)

DISCLAIMER = "Not financial advice. Signals are heuristics, not predictions."
NON_EQUITY_VALUATION = "valuation n/a (no earnings/cash flow)"
STRESS_LABELS = {
    "covid_2020": "COVID crash (19 Feb-23 Mar 2020)",
    "bear_2022": "2022 bear market (3 Jan-12 Oct 2022)",
}
NO_CANDIDATES = "No candidates found (fundamentals unavailable or none passed the filters)"


@dataclass
class HoldingAnalysis:
    """Everything computed for one successfully fetched holding."""

    data: TickerData
    candles: CandleReading
    stop: StopSuggestion
    outlook: Outlook
    sector_median_pe: float | None = None
    has_fundamentals: bool = False


def last_close(data: TickerData) -> float | None:
    prices = data.prices
    if prices is None or "Close" not in prices.columns:
        return None
    close = prices["Close"].dropna()
    if not len(close):
        return None
    value = float(close.iloc[-1])
    return value if math.isfinite(value) else None


def pct_below(price: float | None, stop: float | None) -> str:
    """Percent a stop sits below the price, or 'n/a' when either is missing."""
    if price is None or stop is None or price <= 0:
        return "n/a"
    return f"{(price - stop) / price * 100:.1f}%"


def _stop_line(label: str, price: float | None, stop: float | None) -> str:
    if stop is None:
        return f"- {label}: n/a"
    return f"- {label}: {stop:.2f} ({pct_below(price, stop)} below price)"


def _valuation_line(h: HoldingAnalysis) -> str:
    if h.data.asset_class != "equity":
        return NON_EQUITY_VALUATION
    if not h.has_fundamentals:
        return "valuation skipped (fundamentals unavailable)"
    pe = h.data.fundamentals.get("trailingPE")
    parts = [f"P/E {float(pe):.1f}"] if isinstance(pe, (int, float)) else []
    if h.sector_median_pe is not None:
        parts.append(f"sector median P/E {h.sector_median_pe:.1f}")
    return "Valuation: " + (", ".join(parts) if parts else "no sector comparison available")


def _holding_section(holding: Holding, h: HoldingAnalysis) -> list[str]:
    price = last_close(h.data)
    price_txt = f"{price:.2f}" if price is not None else "n/a"
    lines = [
        f"## {holding.ticker} ({h.data.asset_class}, {holding.weight:.1%} of portfolio)",
        "",
        f"Last close: {price_txt}",
        "",
        "**Bullish**",
        *[f"- {b}" for b in h.outlook.bullish],
        "",
        "**Bearish**",
        *[f"- {b}" for b in h.outlook.bearish],
        "",
    ]
    if h.outlook.narrative:
        lines += ["**Narrative**", "", h.outlook.narrative, ""]
    c = h.candles
    rsi = f"{c.rsi:.1f}" if c.rsi is not None else "n/a"
    lines.append(f"**Candlesticks:** bias {c.bias}, trend {c.trend}, RSI {rsi}")
    if c.patterns:
        lines += [f"- {p.name} ({p.direction}, {p.date:%Y-%m-%d})" for p in c.patterns]
    else:
        lines.append("- No recent patterns")
    if c.note:
        lines.append(f"- Note: {c.note}")
    lines += ["", "**Stop loss**"]
    lines.append(_stop_line("ATR stop (2 x ATR)", price, h.stop.atr_stop))
    lines.append(_stop_line("Support stop", price, h.stop.support_stop))
    if h.stop.recommended is None:
        lines.append("- Recommended: n/a")
    else:
        lines.append(
            f"- Recommended: {h.stop.recommended:.2f} "
            f"({pct_below(price, h.stop.recommended)} below price, {h.stop.method} method)"
        )
    if h.stop.note:
        lines.append(f"- Note: {h.stop.note}")
    lines += ["", _valuation_line(h), ""]
    return lines


def _hedge_section(hedge: HedgeResult | None) -> list[str]:
    lines = ["## Portfolio hedge", ""]
    if hedge is None:
        return [*lines, "Hedge analysis unavailable: not run.", ""]
    if not hedge.stress:  # early-return result: beta/correlation are placeholders
        return [*lines, f"Hedge analysis unavailable: {hedge.reason}", ""]
    lines += [
        f"Hedge verdict: {hedge.verdict} - {hedge.reason}",
        "",
        f"- Beta vs benchmark: {hedge.beta:.2f}",
    ]
    if hedge.high_corr_pairs:
        pairs = ", ".join(f"{a}/{b} ({r:.2f})" for a, b, r in hedge.high_corr_pairs)
        lines.append(f"- Highly correlated pairs: {pairs}")
    else:
        lines.append("- Highly correlated pairs: none")
    if hedge.concentration:
        conc = ", ".join(f"{k} {v:.0%}" for k, v in hedge.concentration.items())
        lines.append(f"- Concentration: {conc}")
    lines.append(f"- Offsetting holdings: {', '.join(hedge.offsets) if hedge.offsets else 'none'}")
    lines += ["", "**Stress tests**"]
    for s in hedge.stress:
        name = STRESS_LABELS.get(s.name, s.name)
        if s.skipped_reason is not None:
            lines.append(f"- {name}: skipped ({s.skipped_reason})")
        else:
            lines.append(
                f"- {name}: portfolio {s.portfolio_return:+.1%} vs benchmark "
                f"{s.benchmark_return:+.1%}"
            )
    return [*lines, ""]


def _underpriced_section(candidates: list[ValuationCandidate], skipped: list[str]) -> list[str]:
    lines = ["## Underpriced candidates (to research, not buy recommendations)", ""]
    if candidates:
        for c in candidates:
            lines.append(f"- {c.ticker} (score {c.score:.2f}): {'; '.join(c.reasons)}")
    else:
        lines.append(NO_CANDIDATES)
    if skipped:
        lines += [
            "",
            "Held equities skipped for missing fundamentals: " + ", ".join(skipped),
        ]
    return [*lines, ""]


def render_markdown(
    *,
    holdings: list[Holding],
    analyses: dict[str, HoldingAnalysis],
    hedge: HedgeResult | None,
    candidates: list[ValuationCandidate],
    errors: dict[str, str],
    skipped_valuation: list[str],
    warnings: list[str],
) -> str:
    lines = ["# Portfolio analysis", "", f"_{DISCLAIMER}_", ""]
    if warnings:
        lines += ["## Warnings", "", *[f"- {w}" for w in warnings], ""]
    if errors:
        lines += ["## Errors", "", *[f"- {t}: {msg}" for t, msg in errors.items()], ""]
    for holding in holdings:
        if holding.ticker in analyses:
            lines += _holding_section(holding, analyses[holding.ticker])
    lines += _hedge_section(hedge)
    lines += _underpriced_section(candidates, skipped_valuation)
    lines += ["---", DISCLAIMER, ""]
    return "\n".join(lines)
