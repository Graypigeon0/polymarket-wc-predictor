from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer import data
from portfolio_analyzer.data import (
    DataError,
    DataProviderError,
    annualisation_factor,
    classify,
    fetch_ticker,
)

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _raw(n: int = 5, tz: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="America/New_York" if tz else None)
    df = pd.DataFrame({c: np.arange(1.0, n + 1) for c in COLS}, index=idx)
    df["Dividends"] = 0.0
    df["Stock Splits"] = 0.0
    return df


class FakeTicker:
    calls: ClassVar[int] = 0
    history_exc: ClassVar[Exception | None] = None
    info_exc: ClassVar[Exception | None] = None
    frame: ClassVar[pd.DataFrame] = _raw()
    info_data: ClassVar[dict] = {"quoteType": "EQUITY", "trailingPE": 20.0, "marketCap": 5}
    meta: ClassVar[dict] = {"instrumentType": "EQUITY"}

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, **kwargs):
        type(self).calls += 1
        if type(self).history_exc is not None:
            raise type(self).history_exc
        return type(self).frame.copy()

    @property
    def info(self):
        if type(self).info_exc is not None:
            raise type(self).info_exc
        return type(self).info_data

    @property
    def history_metadata(self):
        return type(self).meta


@pytest.fixture(autouse=True)
def fake(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return FakeTicker
    FakeTicker.calls = 0
    FakeTicker.history_exc = None
    FakeTicker.info_exc = None
    FakeTicker.frame = _raw()
    FakeTicker.info_data = {"quoteType": "EQUITY", "trailingPE": 20.0, "marketCap": 5}
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


def test_provider_outage_is_distinct_from_missing_ticker(tmp_path, fake):
    fake.history_exc = ConnectionError("proxy refused tunnel")
    with pytest.raises(DataProviderError) as outage:
        fetch_ticker("AAA", cache_dir=tmp_path)
    assert "Could not reach Yahoo Finance" in str(outage.value)
    assert "proxy refused tunnel" in str(outage.value)

    fake.history_exc = None
    fake.frame = pd.DataFrame(columns=COLS)
    with pytest.raises(DataError) as empty:
        fetch_ticker("AAA", cache_dir=tmp_path)
    assert not isinstance(empty.value, DataProviderError)
    assert "No price data" in str(empty.value)
    assert "Could not reach" not in str(empty.value)


def test_yfinance_missing_ticker_error_is_plain_dataerror(tmp_path, fake):
    from yfinance import exceptions as yf_exc

    fake.history_exc = yf_exc.YFTzMissingError("ZZZ")
    with pytest.raises(DataError) as err:
        fetch_ticker("ZZZ", cache_dir=tmp_path)
    assert not isinstance(err.value, DataProviderError)


def test_quote_type_falls_back_to_info(tmp_path, fake):
    fake.meta = {}
    fake.info_data = {"quoteType": "CRYPTOCURRENCY"}
    assert fetch_ticker("XYZ", cache_dir=tmp_path).asset_class == "crypto"


def test_info_failure_gives_empty_fundamentals(tmp_path, fake):
    fake.info_exc = RuntimeError("401 Invalid Crumb")
    td = fetch_ticker("AAA", cache_dir=tmp_path)
    assert td.fundamentals == {}
    assert len(td.prices) == 5
