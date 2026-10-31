from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_analyzer.candles import analyse_candles
from portfolio_analyzer.models import (
    HedgeResult,
    Holding,
    StopSuggestion,
    StressResult,
    TickerData,
    ValuationCandidate,
)
from portfolio_analyzer.outlook import build_outlook
from portfolio_analyzer.report import (
    DISCLAIMER,
    NO_CANDIDATES,
    NON_EQUITY_VALUATION,
    HoldingAnalysis,
    annualised_volatility,
    pct_below,
    render_markdown,
)
from portfolio_analyzer.valuation import ScreenStats


def _td(ticker: str, asset_class: str = "equity", fundamentals=None) -> TickerData:
    rng = np.random.default_rng(abs(hash(ticker)) % 1000)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300))
    idx = pd.bdate_range("2024-01-01", periods=300)
    df = pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1.0},
        index=idx,
    )
    return TickerData(ticker, asset_class, df, fundamentals or {})


def _analysis(ticker="AAA", asset_class="equity", fundamentals=None, has_f=False, med=None):
    td = _td(ticker, asset_class, fundamentals)
    last = float(td.prices["Close"].iloc[-1])
    stop = StopSuggestion(ticker, last * 0.9, None, last * 0.9, 10.0, "atr", None)
    candles = analyse_candles(td)
    return HoldingAnalysis(td, candles, stop, build_outlook(td, candles, stop), med, has_f)


def _hedge(**kw) -> HedgeResult:
    base = dict(
        beta=0.8,
        correlation=pd.DataFrame(),
        high_corr_pairs=[("AAA", "BBB", 0.9)],
        concentration={"equity": 1.0},
        offsets=[],
        stress=[
            StressResult("March 2020", -0.1, -0.3, None),
            StressResult("2022", None, None, "history starts after window"),
        ],
        verdict="partial",
        reason="Portfolio beta is 0.80.",
    )
    base.update(kw)
    return HedgeResult(**base)


def _render(
    analyses, holdings=None, hedge=None, candidates=(), errors=None, skipped=(), warns=(), **kw
):
    holdings = holdings or [Holding(t, 1 / len(analyses)) for t in analyses]
    return render_markdown(
        holdings=holdings,
        analyses=analyses,
        hedge=hedge,
        candidates=list(candidates),
        errors=errors or {},
        skipped_valuation=list(skipped),
        warnings=list(warns),
        **kw,
    )


def test_report_contains_disclaimer():
    md = _render({"AAA": _analysis()})
    assert "Not financial advice. Signals are heuristics, not predictions." in md
    assert DISCLAIMER in md


def test_report_has_section_per_holding_with_bull_bear_candles_stop():
    md = _render({"AAA": _analysis("AAA"), "BBB": _analysis("BBB")})
    for t in ("AAA", "BBB"):
        assert f"## {t} (" in md
    assert md.count("**Bullish**") == 2
    assert md.count("**Bearish**") == 2
    assert md.count("**Candlesticks:**") == 2
    assert md.count("**Stop loss**") == 2


def test_report_lists_failed_ticker_in_errors_and_still_renders_others():
    holdings = [Holding("AAA", 0.5), Holding("BAD", 0.5)]
    md = _render({"AAA": _analysis()}, holdings=holdings, errors={"BAD": "No price data"})
    assert "## Errors" in md
    assert "- BAD: No price data" in md
    assert "## AAA (" in md
    assert "## BAD (" not in md


def test_non_equity_shows_valuation_na():
    md = _render({"BTC-USD": _analysis("BTC-USD", "crypto")})
    assert NON_EQUITY_VALUATION in md
    assert "valuation n/a (no earnings/cash flow)" in md


def test_equity_without_fundamentals_flagged_skipped():
    md = _render({"AAA": _analysis()}, skipped=["AAA"])
    assert "valuation skipped (fundamentals unavailable)" in md
    assert "skipped for missing fundamentals: AAA" in md


def test_equity_with_fundamentals_shows_pe_and_sector_median():
    a = _analysis(fundamentals={"trailingPE": 12.0}, has_f=True, med=20.0)
    md = _render({"AAA": a})
    assert "P/E 12.0" in md
    assert "sector median P/E 20.0" in md


def test_both_stop_methods_show_percent_risk_and_na():
    a = _analysis()
    price = float(a.data.prices["Close"].iloc[-1])
    a.stop = StopSuggestion("AAA", price * 0.9, price * 0.95, price * 0.95, 5.0, "support", None)
    md = _render({"AAA": a})
    assert "ATR stop (2 x ATR)" in md and "(10.0% below price)" in md
    assert "Support stop" in md and "(5.0% below price)" in md
    a.stop = StopSuggestion("AAA", None, None, None, None, "n/a", "short history")
    md = _render({"AAA": a})
    assert "- ATR stop (2 x ATR): n/a" in md
    assert "- Support stop: n/a" in md
    assert "Recommended: n/a" in md


def test_pct_below():
    assert pct_below(100.0, 90.0) == "10.0%"
    assert pct_below(100.0, None) == "n/a"
    assert pct_below(None, 90.0) == "n/a"


def test_hedge_available_renders_beta_pairs_and_stress():
    md = _render({"AAA": _analysis()}, hedge=_hedge())
    assert "Hedge verdict: partial" in md
    assert "Beta vs benchmark: 0.80" in md
    assert "AAA/BBB (0.90)" in md
    assert "March 2020: portfolio -10.0% vs benchmark -30.0%" in md
    assert "2022: skipped (history starts after window)" in md


def test_hedge_unavailable_hides_numbers():
    h = _hedge(stress=[], beta=0.0, reason="No holdings with data to analyse.", high_corr_pairs=[])
    md = _render({"AAA": _analysis()}, hedge=h)
    assert "Hedge analysis unavailable: No holdings with data to analyse." in md
    assert "Beta vs benchmark" not in md
    assert "Hedge verdict" not in md


def test_candidates_and_empty_message():
    c = ValuationCandidate("ZZZ", 0.9, ["low P/E", "high FCF yield"])
    md = _render({"AAA": _analysis()}, candidates=[c])
    assert "ZZZ (score 0.90): low P/E; high FCF yield" in md
    assert NO_CANDIDATES not in md
    md = _render({"AAA": _analysis()})
    assert "No candidates found (fundamentals unavailable or none passed the filters)" in md


def test_warnings_in_report():
    md = _render({"AAA": _analysis()}, warns=["Weights sum to 90%, normalised to 100%"])
    assert "## Warnings" in md
    assert "Weights sum to 90%" in md


def test_stress_windows_use_readable_labels_with_fallback():
    h = _hedge(
        stress=[
            StressResult("covid_2020", -0.1, -0.3, None),
            StressResult("bear_2022", None, None, "no history"),
            StressResult("custom_x", 0.0, 0.0, None),
        ]
    )
    md = _render({"AAA": _analysis()}, hedge=h)
    assert "COVID crash (19 Feb-23 Mar 2020): portfolio -10.0%" in md
    assert "2022 bear market (3 Jan-12 Oct 2022): skipped (no history)" in md
    assert "custom_x: portfolio" in md
    assert "covid_2020" not in md


def test_stats_line_under_candidates_and_no_candidates_message():
    stats = ScreenStats(considered=100, skipped_fundamentals=10, excluded_by_filters=40,
                        small_sectors=["Energy", "Utilities"])
    md = _render({"AAA": _analysis()}, screen_stats=stats)
    expect = (
        "Universe: 100 equities fetched; 10 skipped (missing fundamentals); 40 excluded by "
        "filters; sectors with fewer than 3 qualifying peers left out: Energy, Utilities."
    )
    assert expect in md
    assert md.index(NO_CANDIDATES) < md.index(expect)
    assert "unreliable" not in md
    c = ValuationCandidate("ZZZ", 0.9, ["low P/E"])
    md = _render({"AAA": _analysis()}, candidates=[c], screen_stats=stats)
    assert md.index("ZZZ (score") < md.index(expect)


def test_stats_line_with_no_small_sectors_says_none():
    md = _render({"AAA": _analysis()}, screen_stats=ScreenStats(10, 1, 2, []))
    assert "left out: none." in md


def test_warning_when_more_than_half_skipped_for_fundamentals():
    md = _render({"AAA": _analysis()}, screen_stats=ScreenStats(500, 401, 10, []))
    assert "WARNING" in md and "fundamentals were largely unavailable" in md
    assert "unreliable" in md
    md = _render({"AAA": _analysis()}, screen_stats=ScreenStats(500, 250, 10, []))
    assert "WARNING" not in md  # exactly half is not "more than half"


def test_screen_skipped_message():
    md = _render({"AAA": _analysis()}, screen_skipped=True)
    assert "Underpriced screen skipped (--universe none)" in md
    assert NO_CANDIDATES not in md


def test_annualised_volatility_uses_365_for_crypto_and_252_otherwise():
    eq = _td("AAA", "equity")
    cr = TickerData("BTC-USD", "crypto", eq.prices.copy(), {})
    rets = eq.prices["Close"].pct_change().dropna()
    expect_eq = float(rets.std() * np.sqrt(252) * 100)
    expect_cr = float(rets.std() * np.sqrt(365) * 100)
    assert abs(annualised_volatility(eq) - expect_eq) < 1e-9
    assert abs(annualised_volatility(cr) - expect_cr) < 1e-9
    assert annualised_volatility(cr) > annualised_volatility(eq)
    for ac in ("etf", "commodity"):
        assert abs(annualised_volatility(TickerData("X", ac, eq.prices, {})) - expect_eq) < 1e-9


def test_annualised_volatility_none_with_fewer_than_30_returns():
    td = _td("AAA")
    short = TickerData("AAA", "equity", td.prices.iloc[:30], {})  # 29 returns
    assert annualised_volatility(short) is None
    ok = TickerData("AAA", "equity", td.prices.iloc[:31], {})  # 30 returns
    assert annualised_volatility(ok) is not None


def test_report_shows_annualised_volatility_line():
    a = _analysis("AAA")
    md = _render({"AAA": a})
    vol = annualised_volatility(a.data)
    assert f"Annualised volatility: {vol:.1f}%" in md
    short = _analysis("SHORT")
    short.data = TickerData("SHORT", "equity", short.data.prices.iloc[:20], {})
    assert "Annualised volatility: n/a" in _render({"SHORT": short})


def _oil_data(n=60, seed=3):
    rng = np.random.default_rng(seed)
    close = pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.02, n)),
                      index=pd.bdate_range("2020-03-02", periods=n))
    close.iloc[20] = -10.0
    close.iloc[21] = 8.0
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0})
    return TickerData("CL=F", "commodity", df, {})


def test_volatility_excludes_non_positive_price_returns():
    td = _oil_data()
    close = td.prices["Close"]
    rets = close.pct_change(fill_method=None)
    bad = (close <= 0) | (close.shift(1) <= 0)
    expect = float(rets[~bad].dropna().std() * np.sqrt(252) * 100)
    assert abs(annualised_volatility(td) - expect) < 1e-9
    a = _analysis("CL=F", "commodity")
    a.data = td
    md = _render({"CL=F": a})
    note = "(excludes 2 returns across non-positive prices)"
    assert f"Annualised volatility: {expect:.1f}% {note}" in md


def test_ordinary_volatility_has_no_exclusion_note():
    md = _render({"AAA": _analysis("AAA")})
    assert "excludes" not in md.split("Annualised volatility")[1].split("\n")[0]
