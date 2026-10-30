from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer import data
from portfolio_analyzer.data import DataError, annualisation_factor, classify, fetch_ticker

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _raw(n: int = 5, tz: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="America/New_York" if tz else None)
    df = pd.DataFrame({c: np.arange(1.0, n + 1) for c in COLS}, index=idx)
    df["Dividends"] = 0.0
    df["Stock Splits"] = 0.0
    return df


class FakeTicker:
    calls: ClassVar[int] = 0
    frame: ClassVar[pd.DataFrame] = _raw()
    info: ClassVar[dict] = {"quoteType": "EQUITY", "trailingPE": 20.0, "marketCap": 5}
    meta: ClassVar[dict] = {"instrumentType": "EQUITY"}

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, **kwargs):
        type(self).calls += 1
        return type(self).frame.copy()

    @property
    def history_metadata(self):
        return type(self).meta


@pytest.fixture(autouse=True)
def fake(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return FakeTicker
    FakeTicker.calls = 0
    FakeTicker.frame = _raw()
    FakeTicker.info = {"quoteType": "EQUITY", "trailingPE": 20.0, "marketCap": 5}
    FakeTicker.meta = {"instrumentType": "EQUITY"}
    monkeypatch.setattr(data.yf, "Ticker", FakeTicker)
    return FakeTicker


def test_classify_quote_types():
    assert classify("CRYPTOCURRENCY", "X") == "crypto"
    assert classify("FUTURE", "X") == "commodity"
    assert classify("ETF", "X") == "etf"
    assert classify("EQUITY", "X") == "equity"


def test_classify_suffix_fallbacks():
    assert classify("", "CL=F") == "commodity"
    assert classify("MUTUALFUND", "ETH-USD") == "crypto"
    assert classify("", "AAPL") == "equity"


def test_annualisation_factor():
    assert annualisation_factor("crypto") == 365
    for ac in ("equity", "etf", "commodity"):
        assert annualisation_factor(ac) == 252


def test_fetch_drops_nan_rows_and_sorts_index(tmp_path, fake):
    df = _raw(5)
    df.iloc[2, df.columns.get_loc("Close")] = np.nan
    fake.frame = df.iloc[::-1]
    td = fetch_ticker("AAA", cache_dir=tmp_path)
    assert list(td.prices.columns) == COLS
    assert len(td.prices) == 4
    assert td.prices.index.is_monotonic_increasing
    assert td.prices.index.tz is None
    assert td.asset_class == "equity"
    assert td.fundamentals == {"trailingPE": 20.0, "marketCap": 5}


def test_fetch_empty_raises_dataerror(tmp_path, fake):
    fake.frame = pd.DataFrame(columns=COLS)
    with pytest.raises(DataError):
        fetch_ticker("BAD", cache_dir=tmp_path)


def test_cache_hit_skips_second_download(tmp_path, fake):
    first = fetch_ticker("AAA", cache_dir=tmp_path)
    assert fake.calls == 1
    second = fetch_ticker("AAA", cache_dir=tmp_path)
    assert fake.calls == 1
    pd.testing.assert_frame_equal(first.prices, second.prices)
    assert second.asset_class == first.asset_class
    assert second.fundamentals == first.fundamentals


@pytest.mark.network
def test_live_fetch_spy(tmp_path):
    td = fetch_ticker("SPY", cache_dir=tmp_path)
    assert len(td.prices) >= 200
    assert list(td.prices.columns) == COLS
    assert td.asset_class == "etf"
