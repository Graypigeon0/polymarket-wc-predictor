from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_analyzer import valuation
from portfolio_analyzer.models import TickerData
from portfolio_analyzer.valuation import (
    MIN_SECTOR_PEERS,
    ScreenStats,
    has_fundamentals,
    load_universe,
    screen_underpriced,
    screen_with_stats,
)


def _prices(n: int = 260, last: float | None = None) -> pd.DataFrame:
    close = np.full(n, 100.0)
    if last is not None:
        close[-1] = last
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000.0},
        index=idx,
    )


def _eq(ticker="AAA", pe=15.0, fcf=1e9, cap=2e10, sector="Tech", prices=None, **extra):
    f = {"trailingPE": pe, "freeCashflow": fcf, "marketCap": cap, "sector": sector}
    f.update(extra)
    f = {k: v for k, v in f.items() if v is not None}
    return TickerData(ticker, "equity", _prices() if prices is None else prices, f)


def test_non_equity_excluded():
    td = _eq("ETF1")
    td.asset_class = "etf"
    assert screen_underpriced({"ETF1": td}) == []


def test_negative_earnings_excluded():
    u = {"NEG": _eq("NEG", pe=-5.0), "ZERO": _eq("ZERO", pe=0.0), "NOFCF": _eq("NOFCF", fcf=-1.0)}
    assert screen_underpriced(u) == []


def test_broken_trend_excluded():
    # last close 80 < 0.9 * SMA200 (~99.9)
    assert screen_underpriced({"BRK": _eq("BRK", prices=_prices(last=80.0))}) == []


def test_short_history_excluded_without_crash():
    # Fewer than 200 rows: SMA200 undefined, so the stock fails the trend filter (excluded).
    assert screen_underpriced({"NEW": _eq("NEW", prices=_prices(150))}) == []


def test_cheaper_stock_ranks_higher_within_sector():
    u = {
        "CHEAP": _eq("CHEAP", pe=8.0, fcf=3e9, cap=2e10, pegRatio=0.8),
        "MID": _eq("MID", pe=15.0, fcf=2e9, cap=2e10, pegRatio=1.5),
        "RICH": _eq("RICH", pe=30.0, fcf=1e9, cap=2e10, pegRatio=3.0),
    }
    out = screen_underpriced(u)
    assert [c.ticker for c in out] == ["CHEAP", "MID", "RICH"]
    assert out[0].score > out[1].score > out[2].score
    assert any(r.startswith("P/E 8.0 vs sector median 15.0") for r in out[0].reasons)


def test_missing_peg_still_scored():
    u = {
        "A": _eq("A", pe=10.0),
        "B": _eq("B", pe=20.0, pegRatio=2.0),
        "C": _eq("C", pe=30.0, pegRatio=3.0),
    }
    out = screen_underpriced(u)
    assert {c.ticker for c in out} == {"A", "B", "C"}
    assert all(0.0 <= c.score <= 1.0 for c in out)


def test_missing_fundamentals_skipped_not_crashing():
    bare = TickerData("BARE", "equity", _prices(), {})
    nan_cap = _eq("NANCAP", cap=float("nan"))
    zero_cap = _eq("ZCAP", cap=0.0)
    assert screen_underpriced({"BARE": bare}) == []
    out = screen_underpriced({"NANCAP": nan_cap, "ZCAP": zero_cap})
    assert out == [] or all(np.isfinite(c.score) for c in out)


def test_unknown_sector_peers_ranked_together_and_scores_finite():
    u = {f"N{i}": _eq(f"N{i}", pe=10.0 + i, sector=None) for i in range(3)}
    out = screen_underpriced(u)
    assert [c.ticker for c in out] == ["N0", "N1", "N2"]
    assert all(np.isfinite(c.score) for c in out)


def test_sector_below_min_peers_yields_nothing():
    assert MIN_SECTOR_PEERS == 3
    two = {"A": _eq("A", pe=5.0, sector="Energy"), "B": _eq("B", pe=9.0, sector="Energy")}
    assert screen_underpriced(two) == []
    solo = {"SOLO": _eq("SOLO", sector="Utilities")}
    assert screen_underpriced(solo) == []
    three = {**two, "C": _eq("C", pe=12.0, sector="Energy")}
    assert [c.ticker for c in screen_underpriced(three)] == ["A", "B", "C"]


def test_small_sector_does_not_outrank_large_sector_candidates():
    u = {f"T{i}": _eq(f"T{i}", pe=10.0 + i) for i in range(60)}
    u["E1"] = _eq("E1", pe=4.0, sector="Energy")
    u["E2"] = _eq("E2", pe=40.0, sector="Energy")
    out = screen_underpriced(u, top=100)
    assert "E1" not in {c.ticker for c in out} and "E2" not in {c.ticker for c in out}
    assert len(out) == 60


def test_reason_includes_sector_peer_count():
    u = {f"T{i}": _eq(f"T{i}", pe=10.0 + i) for i in range(5)}
    out = screen_underpriced(u)
    assert out[0].reasons[0] == "P/E 10.0 vs sector median 12.0 (n=5)"


def test_screen_with_stats_counts_everything():
    etf = _eq("ETF1")
    etf.asset_class = "etf"
    u = {
        "ETF1": etf,  # not an equity: not counted
        "BARE": TickerData("BARE", "equity", _prices(), {}),  # missing fundamentals
        "NEG": _eq("NEG", pe=-5.0),  # filter: P/E
        "NOFCF": _eq("NOFCF", fcf=-1.0),  # filter: FCF
        "BRK": _eq("BRK", prices=_prices(last=80.0)),  # filter: trend
        "T0": _eq("T0", pe=10.0),
        "T1": _eq("T1", pe=11.0),
        "T2": _eq("T2", pe=12.0),
        "E1": _eq("E1", pe=9.0, sector="Energy"),
        "E2": _eq("E2", pe=19.0, sector="Energy"),
    }
    cands, stats = screen_with_stats(u, top=10)
    assert isinstance(stats, ScreenStats)
    assert stats.considered == 9
    assert stats.skipped_fundamentals == 1
    assert stats.excluded_by_filters == 3
    assert stats.small_sectors == ["Energy"]
    assert [c.ticker for c in cands] == ["T0", "T1", "T2"]
    assert cands == screen_underpriced(u, top=10)


def test_screen_with_stats_empty_universe():
    cands, stats = screen_with_stats({})
    assert cands == [] and stats == ScreenStats(0, 0, 0, [])


def test_52_week_range_fallback_from_prices():
    px = _prices()
    px.iloc[-1, px.columns.get_loc("Close")] = 100.0
    td = _eq("A", prices=px)  # no fiftyTwo* keys: computed from last 252 rows
    td2 = _eq("B", fiftyTwoWeekHigh=120.0, fiftyTwoWeekLow=80.0)
    out = screen_underpriced({"A": td, "B": td2, "C": _eq("C", pe=30.0)})
    by = {c.ticker: c for c in out}
    assert any("52-week range" in r for r in by["A"].reasons)
    assert any("52-week range" in r for r in by["B"].reasons)


def test_top_limits_results():
    u = {f"T{i}": _eq(f"T{i}", pe=10.0 + i) for i in range(6)}
    assert len(screen_underpriced(u, top=3)) == 3
    assert screen_underpriced(u, top=3)[0].ticker == "T0"


def test_has_fundamentals():
    assert has_fundamentals(_eq())
    assert not has_fundamentals(TickerData("X", "equity", _prices(), {"trailingPE": 10.0}))
    assert not has_fundamentals(_eq(cap=None))
    assert not has_fundamentals(_eq(cap=float("nan")))
    etf = _eq()
    etf.asset_class = "etf"
    assert not has_fundamentals(etf)


def test_load_universe_from_file(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("# comment\naapl\n\n  msft  \n#x\nbrk-b\n")
    assert load_universe(str(p)) == ["AAPL", "MSFT", "BRK-B"]


def test_load_universe_sp500_replaces_dots(monkeypatch):
    table = pd.DataFrame({"Symbol": ["AAPL", "BRK.B", "BF.B"]})
    monkeypatch.setattr(valuation.pd, "read_html", lambda *a, **k: [table])
    assert load_universe("sp500") == ["AAPL", "BRK-B", "BF-B"]
