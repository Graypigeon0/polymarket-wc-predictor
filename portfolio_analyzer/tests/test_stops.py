from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer.models import TickerData
from portfolio_analyzer.stops import atr, suggest_stop


def _df(close, half_range: float = 2.5, lows: dict[int, float] | None = None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    low = close - half_range
    for i, v in (lows or {}).items():
        low[i] = v
    idx = pd.bdate_range("2024-01-01", periods=len(close))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + half_range,
            "Low": low,
            "Close": close,
            "Volume": 1000.0,
        },
        index=idx,
    )


def _td(df: pd.DataFrame, asset_class="equity") -> TickerData:
    return TickerData("TEST", asset_class, df, {})


def _dip(low: float) -> TickerData:
    """Flat 100 close, range 5 (ATR ~5), with one wick down to `low` at bar 40."""
    return _td(_df([100.0] * 80, lows={40: low}))


def test_atr_known_values():
    df = _df([100.0] * 30, half_range=1.0)  # constant range 2
    out = atr(df)
    assert out.iloc[-1] == pytest.approx(2.0)


def test_atr_stop_is_price_minus_two_atr():
    td = _td(_df(np.linspace(100, 160, 80), half_range=1.0))
    s = suggest_stop(td)
    expected = 160.0 - 2 * atr(td.prices).iloc[-1]
    assert s.atr_stop == pytest.approx(expected)
    assert s.support_stop is None  # steadily rising: no swing-low


def test_support_stop_below_swing_low():
    # V-shaped: 110 down to 90 at bar 50, back up to 110, then flat
    close = np.concatenate(
        [np.full(30, 110.0), np.linspace(110, 90, 21)[1:], np.linspace(90, 110, 21)[1:]]
    )
    close = np.concatenate([close, np.full(10, 110.0)])
    df = _df(close, half_range=0.5)
    s = suggest_stop(_td(df))
    swing_low = df["Low"].iloc[49]  # trough bar (close 90)
    assert df["Low"].min() == swing_low
    assert s.support_stop == pytest.approx(swing_low * 0.99)


def test_recommends_tighter_when_beyond_one_atr():
    s = suggest_stop(_dip(94.0))
    a = atr(_dip(94.0).prices).iloc[-1]
    assert s.support_stop == pytest.approx(94.0 * 0.99)
    assert s.atr_stop < s.support_stop < 100.0 - a
    assert s.method == "support"
    assert s.recommended == pytest.approx(s.support_stop)
    assert s.risk_pct == pytest.approx((100.0 - s.recommended) / 100.0 * 100)


def test_falls_back_to_atr_when_support_within_one_atr():
    s = suggest_stop(_dip(97.0))
    assert s.support_stop == pytest.approx(97.0 * 0.99)
    assert s.method == "atr"
    assert s.recommended == pytest.approx(s.atr_stop)


def test_no_swing_low_uses_atr_only():
    s = suggest_stop(_td(_df(np.linspace(100, 200, 80), half_range=1.0)))
    assert s.support_stop is None
    assert s.method == "atr"
    assert s.recommended == pytest.approx(s.atr_stop)
    assert s.note is not None and "swing" in s.note.lower()


def test_under_60_bars_returns_na_with_note():
    s = suggest_stop(_td(_df([100.0] * 59)))
    assert s.method == "n/a"
    assert "insufficient data" in s.note
    assert s.atr_stop is None and s.support_stop is None
    assert s.recommended is None and s.risk_pct is None


def test_flat_zero_volatility_does_not_crash():
    s = suggest_stop(_td(_df([100.0] * 80, half_range=0.0)))
    assert s.method == "atr"
    assert s.atr_stop == pytest.approx(100.0)
    assert s.recommended == pytest.approx(100.0)
    assert s.risk_pct == 0.0
    assert "zero volatility" in s.note


def test_crypto_wider_stop_than_stable_series():
    sign = np.where(np.arange(120) % 2 == 0, 1.0, -1.0)
    stable = _td(_df(100 + 0.5 * sign, half_range=0.2), "etf")
    wild = _td(_df(100 + 10.0 * sign, half_range=3.0), "crypto")
    s_stable, s_wild = suggest_stop(stable), suggest_stop(wild)
    assert s_stable.risk_pct is not None and s_wild.risk_pct is not None
    assert s_wild.risk_pct > s_stable.risk_pct


def test_non_positive_atr_stop_no_swing_low_is_na():
    s = suggest_stop(_td(_df([1.0] * 80, half_range=0.6)))  # ATR 1.2 > price / 2
    assert s.atr_stop is None
    assert s.recommended is None and s.risk_pct is None
    assert s.method == "n/a"
    assert "volatil" in s.note


def test_non_positive_atr_stop_with_support_recommends_support():
    s = suggest_stop(_td(_df([1.0] * 80, half_range=0.6, lows={40: 0.3})))
    assert s.atr_stop is None
    assert s.support_stop == pytest.approx(0.3 * 0.99)
    assert s.method == "support"
    assert s.recommended == pytest.approx(s.support_stop)
    assert 0 < s.risk_pct < 100
    assert "volatil" in s.note
