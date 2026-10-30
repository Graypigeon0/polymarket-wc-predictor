from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime
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
    info_calls: ClassVar[int] = 0
    meta_calls: ClassVar[int] = 0
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
        type(self).info_calls += 1
        if type(self).info_exc is not None:
            raise type(self).info_exc
        return type(self).info_data

    @property
    def history_metadata(self):
        type(self).meta_calls += 1
        return type(self).meta


@pytest.fixture(autouse=True)
def fake(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return FakeTicker
    FakeTicker.calls = 0
    FakeTicker.info_calls = 0
    FakeTicker.meta_calls = 0
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


# ---- I2: transient info failure must not be cached for the day ----------------------------


def test_info_failure_retried_on_cache_hit_without_redownloading_prices(tmp_path, fake):
    fake.info_exc = RuntimeError("401 Invalid Crumb")
    first = fetch_ticker("AAA", cache_dir=tmp_path)
    assert first.fundamentals == {}
    fake.info_exc = None
    second = fetch_ticker("AAA", cache_dir=tmp_path)
    assert fake.calls == 1  # prices not re-downloaded
    assert fake.info_calls == 2
    assert second.fundamentals == {"trailingPE": 20.0, "marketCap": 5}
    # and the success was persisted: a third call makes no further info lookups
    third = fetch_ticker("AAA", cache_dir=tmp_path)
    assert fake.info_calls == 2
    assert third.fundamentals == second.fundamentals


def test_info_retry_failure_is_tolerated_and_cache_not_poisoned(tmp_path, fake):
    fake.info_exc = RuntimeError("429")
    fetch_ticker("AAA", cache_dir=tmp_path)
    again = fetch_ticker("AAA", cache_dir=tmp_path)  # still failing
    assert again.fundamentals == {} and len(again.prices) == 5
    fake.info_exc = None
    assert fetch_ticker("AAA", cache_dir=tmp_path).fundamentals["trailingPE"] == 20.0


def test_unresolved_quote_type_is_retried_not_cached_as_equity(tmp_path, fake):
    fake.meta = {}
    fake.info_exc = RuntimeError("401 Invalid Crumb")
    first = fetch_ticker("SPY", cache_dir=tmp_path)
    assert first.asset_class == "equity"  # suffix fallback for this run only
    fake.meta = {"instrumentType": "ETF"}
    fake.info_exc = None
    second = fetch_ticker("SPY", cache_dir=tmp_path)
    assert second.asset_class == "etf"
    assert fake.calls == 1
    assert fetch_ticker("SPY", cache_dir=tmp_path).asset_class == "etf"


def test_resolved_and_ok_cache_hit_makes_no_extra_calls(tmp_path, fake):
    fetch_ticker("AAA", cache_dir=tmp_path)
    info_calls, meta_calls = fake.info_calls, fake.meta_calls
    fetch_ticker("AAA", cache_dir=tmp_path)
    assert (fake.calls, fake.info_calls, fake.meta_calls) == (1, info_calls, meta_calls)


# ---- I5: no pickle, CSV + JSON cache -------------------------------------------------------


def test_cache_uses_csv_and_json_never_pickle(tmp_path, fake):
    fetch_ticker("AAA", cache_dir=tmp_path)
    suffixes = sorted(p.suffix for p in tmp_path.iterdir())
    assert suffixes == [".csv", ".json"]
    assert "pickle" not in inspect.getsource(data).lower()
    meta = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert meta["info_ok"] is True and meta["type_resolved"] is True


def test_csv_round_trip_preserves_index_and_dtypes(tmp_path, fake):
    fresh = fetch_ticker("AAA", cache_dir=tmp_path)
    cached = fetch_ticker("AAA", cache_dir=tmp_path)
    assert fake.calls == 1
    pd.testing.assert_frame_equal(fresh.prices, cached.prices)
    assert cached.prices.index.tz is None
    assert isinstance(cached.prices.index, pd.DatetimeIndex)
    assert all(dt == np.float64 for dt in cached.prices.dtypes)


def test_corrupt_or_old_format_cache_is_refetched(tmp_path, fake):
    fetch_ticker("AAA", cache_dir=tmp_path)
    next(tmp_path.glob("*.csv")).write_text("garbage,,\n\x00")
    fetch_ticker("AAA", cache_dir=tmp_path)
    assert fake.calls == 2
    meta_path = next(tmp_path.glob("*.json"))
    meta_path.write_text(json.dumps({"ticker": "AAA", "asset_class": "equity"}))  # old format
    fetch_ticker("AAA", cache_dir=tmp_path)
    assert fake.calls == 3


def test_default_cache_dir_is_private_under_home(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(data.Path, "home", classmethod(lambda cls: tmp_path))
    fetch_ticker("AAA")
    d = tmp_path / ".cache" / "portfolio_analyzer"
    assert d.is_dir()
    assert (d.stat().st_mode & 0o777) == 0o700
    assert any(d.glob("*.csv"))


# ---- I6: drop today's unfinished bar -------------------------------------------------------


def _utc(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=UTC)


def test_last_row_dated_today_is_dropped(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(data, "_now", lambda: _utc(2024, 1, 5, 18))  # 13:00 New York
    td = fetch_ticker("AAA", cache_dir=tmp_path)
    assert len(td.prices) == 4
    assert td.prices.index.max() == pd.Timestamp("2024-01-04")


def test_last_row_dated_yesterday_is_kept(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(data, "_now", lambda: _utc(2024, 1, 6, 18))
    assert len(fetch_ticker("AAA", cache_dir=tmp_path).prices) == 5


def test_today_is_judged_in_the_exchange_time_zone(tmp_path, fake, monkeypatch):
    # 02:00 UTC on the 6th is still the 5th in New York: the 5th bar is not finished
    monkeypatch.setattr(data, "_now", lambda: _utc(2024, 1, 6, 2))
    assert len(fetch_ticker("AAA", cache_dir=tmp_path).prices) == 4


def test_crypto_utc_today_bar_dropped(tmp_path, fake, monkeypatch):
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({c: np.arange(1.0, 6) for c in COLS}, index=idx)
    fake.frame = df
    fake.meta = {"instrumentType": "CRYPTOCURRENCY"}
    monkeypatch.setattr(data, "_now", lambda: _utc(2024, 1, 5, 23))
    assert len(fetch_ticker("BTC-USD", cache_dir=tmp_path).prices) == 4
    monkeypatch.setattr(data, "_now", lambda: _utc(2024, 1, 6, 0))
    assert len(fetch_ticker("BTC-USD", start="2019-12-02", cache_dir=tmp_path).prices) == 5


# ---- I7: API drift is not an outage --------------------------------------------------------


def test_type_error_is_not_reported_as_yahoo_outage(tmp_path, fake):
    fake.history_exc = TypeError("history() got an unexpected keyword argument 'raise_errors'")
    with pytest.raises(RuntimeError) as err:
        fetch_ticker("AAA", cache_dir=tmp_path)
    assert not isinstance(err.value, DataError)
    assert "yfinance API changed" in str(err.value)
    assert "Could not reach" not in str(err.value)
