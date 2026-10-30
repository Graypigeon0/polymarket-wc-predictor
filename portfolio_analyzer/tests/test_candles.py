from __future__ import annotations

import pandas as pd

from portfolio_analyzer.candles import analyse_candles, rsi
from portfolio_analyzer.models import TickerData


def _td(rows: list[tuple[float, float, float, float]]) -> TickerData:
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = 1000.0
    return TickerData("TEST", "equity", df, {})


def _alternating(n: int) -> list[tuple[float, float, float, float]]:
    """Equal-bodied alternating bull/bear bars: no pattern, sideways."""
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append((100.0, 101.5, 99.5, 101.0))
        else:
            rows.append((101.0, 101.5, 99.5, 100.0))
    return rows


def _trending(n: int, start: float, step: float) -> list[tuple[float, float, float, float]]:
    rows = []
    price = start
    for _ in range(n):
        nxt = price + step
        rows.append((price, max(price, nxt) + 0.5, min(price, nxt) - 0.5, nxt))
        price = nxt
    return rows


def _names(reading) -> set[str]:
    return {p.name for p in reading.patterns}


def test_detects_hammer_after_downtrend():
    rows = _trending(60, 200.0, -1.0)
    rows.append((141.0, 141.6, 136.0, 141.5))
    r = analyse_candles(_td(rows))
    assert "hammer" in _names(r)
    assert next(p for p in r.patterns if p.name == "hammer").direction == "bullish"


def test_hammer_not_flagged_in_uptrend():
    rows = _trending(60, 100.0, 1.0)
    last = rows[-1][3]
    rows.append((last, last + 0.6, last - 5.0, last + 0.5))
    assert "hammer" not in _names(analyse_candles(_td(rows)))


def test_detects_shooting_star_after_uptrend():
    rows = _trending(60, 100.0, 1.0)
    last = rows[-1][3]
    rows.append((last, last + 5.0, last - 0.1, last - 0.5))
    r = analyse_candles(_td(rows))
    assert "shooting star" in _names(r)


def test_detects_bullish_engulfing():
    rows = _alternating(60)
    rows.append((105.0, 105.5, 102.5, 103.0))
    rows.append((102.5, 106.0, 102.0, 105.5))
    r = analyse_candles(_td(rows))
    assert "bullish engulfing" in _names(r)
    assert r.bias == "bullish"


def test_detects_bearish_engulfing():
    rows = _alternating(60)
    rows.append((103.0, 105.5, 102.5, 105.0))
    rows.append((105.5, 106.0, 102.0, 102.5))
    assert "bearish engulfing" in _names(analyse_candles(_td(rows)))


def test_detects_evening_star():
    rows = _alternating(60)
    rows.append((100.5, 110.5, 100.0, 110.0))
    rows.append((110.5, 111.5, 110.0, 111.0))
    rows.append((110.0, 110.5, 101.5, 102.0))
    r = analyse_candles(_td(rows))
    assert "evening star" in _names(r)
    assert r.bias == "bearish"


def test_detects_morning_star():
    rows = _alternating(60)
    rows.append((110.0, 110.5, 99.5, 100.0))
    rows.append((99.5, 100.0, 98.5, 99.0))
    rows.append((100.0, 109.5, 99.5, 109.0))
    assert "morning star" in _names(analyse_candles(_td(rows)))


def test_detects_three_white_soldiers():
    rows = _alternating(60)
    rows.append((100.0, 102.5, 99.5, 102.0))
    rows.append((101.5, 104.5, 101.0, 104.0))
    rows.append((103.5, 106.5, 103.0, 106.0))
    r = analyse_candles(_td(rows))
    assert "three white soldiers" in _names(r)
    assert r.bias == "bullish"


def test_detects_three_black_crows():
    rows = _alternating(60)
    rows.append((106.0, 106.5, 103.5, 104.0))
    rows.append((104.5, 105.0, 101.5, 102.0))
    rows.append((102.5, 103.0, 99.5, 100.0))
    r = analyse_candles(_td(rows))
    assert "three black crows" in _names(r)
    assert r.bias == "bearish"


def test_doji():
    rows = _alternating(60)
    rows.append((100.0, 101.0, 99.0, 100.05))
    r = analyse_candles(_td(rows))
    assert "doji" in _names(r)
    assert next(p for p in r.patterns if p.name == "doji").direction == "neutral"


def test_no_patterns_is_neutral():
    r = analyse_candles(_td(_alternating(70)))
    assert r.patterns == []
    assert r.bias == "neutral"
    assert r.trend == "sideways"
    assert r.note is None


def test_trend_up_down_sideways():
    assert analyse_candles(_td(_trending(80, 100.0, 1.0))).trend == "up"
    assert analyse_candles(_td(_trending(80, 200.0, -1.0))).trend == "down"
    assert analyse_candles(_td(_alternating(80))).trend == "sideways"


def test_rsi_all_gains_is_100():
    assert rsi(pd.Series([float(i) for i in range(1, 31)])) == 100.0


def test_rsi_all_losses_is_0():
    assert rsi(pd.Series([float(i) for i in range(30, 0, -1)])) == 0.0


def test_rsi_too_short_is_none():
    assert rsi(pd.Series([1.0, 2.0, 3.0])) is None


def test_flat_prices_rsi_is_none_not_crash():
    assert rsi(pd.Series([100.0] * 40)) is None
    rows = [(100.0, 100.0, 100.0, 100.0)] * 80  # zero-range bars: high == low
    r = analyse_candles(_td(rows))
    assert r.rsi is None
    assert r.patterns == []
    assert r.bias == "neutral"
    assert r.trend == "sideways"


def test_under_60_bars_returns_insufficient_data_note():
    r = analyse_candles(_td(_alternating(59)))
    assert r.bias == "neutral"
    assert r.patterns == []
    assert r.note is not None and "insufficient data" in r.note


def test_empty_and_single_bar_frames():
    empty = TickerData("X", "equity", pd.DataFrame(columns=["Open", "High", "Low", "Close"]), {})
    for td in (empty, _td([(100.0, 101.0, 99.0, 100.5)])):
        r = analyse_candles(td)
        assert r.bias == "neutral"
        assert r.rsi is None
        assert "insufficient data" in (r.note or "")
