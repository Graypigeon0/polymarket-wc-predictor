from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_analyzer import valuation
from portfolio_analyzer.models import TickerData
from portfolio_analyzer.valuation import has_fundamentals, load_universe, screen_underpriced


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
    u = {"A": _eq("A", pe=10.0), "B": _eq("B", pe=20.0, pegRatio=2.0)}
    out = screen_underpriced(u)
    assert {c.ticker for c in out} == {"A", "B"}
    assert all(0.0 <= c.score <= 1.0 for c in out)


def test_missing_fundamentals_skipped_not_crashing():
    bare = TickerData("BARE", "equity", _prices(), {})
    nan_cap = _eq("NANCAP", cap=float("nan"))
    zero_cap = _eq("ZCAP", cap=0.0)
    assert screen_underpriced({"BARE": bare}) == []
    out = screen_underpriced({"NANCAP": nan_cap, "ZCAP": zero_cap})
    assert out == [] or all(np.isfinite(c.score) for c in out)


def test_single_stock_sector_and_unknown_sector_not_nan():
    u = {"SOLO": _eq("SOLO", sector="Utilities"), "NOSEC": _eq("NOSEC", sector=None)}
    out = screen_underpriced(u)
    assert len(out) == 2
    assert all(np.isfinite(c.score) for c in out)


def test_52_week_range_fallback_from_prices():
    px = _prices()
    px.iloc[-1, px.columns.get_loc("Close")] = 100.0
    td = _eq("A", prices=px)  # no fiftyTwo* keys: computed from last 252 rows
    assert len(screen_underpriced({"A": td})) == 1
    td2 = _eq("B", fiftyTwoWeekHigh=120.0, fiftyTwoWeekLow=80.0)
    assert len(screen_underpriced({"B": td2})) == 1


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
