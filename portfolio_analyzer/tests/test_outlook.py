from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer.models import CandleReading, Outlook, StopSuggestion, TickerData
from portfolio_analyzer.outlook import NarrationError, build_outlook, narrate


def _prices(close) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range("2024-01-01", periods=len(close))
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}, index=idx
    )


def _flat_mid() -> pd.DataFrame:
    # range 50..150, last price 100: nowhere near either extreme
    c = np.full(260, 100.0)
    c[10], c[20] = 50.0, 150.0
    return _prices(c)


def _td(asset_class="equity", fundamentals=None, prices=None) -> TickerData:
    return TickerData("TEST", asset_class, _flat_mid() if prices is None else prices,
                      fundamentals or {})


def _c(bias="neutral", trend="sideways", rsi=50.0, note=None) -> CandleReading:
    return CandleReading("TEST", [], bias, trend, rsi, note)


def _s(risk=5.0, note=None) -> StopSuggestion:
    return StopSuggestion("TEST", 95.0, None, 95.0, risk, "atr", note)


def test_oversold_adds_bullish_bullet_with_rsi_value():
    o = build_outlook(_td(), _c(rsi=24.3), _s())
    assert any("24.3" in b and "oversold" in b.lower() for b in o.bullish)


def test_overbought_adds_bearish_bullet():
    o = build_outlook(_td(), _c(rsi=78.0), _s())
    assert any("78.0" in b and "overbought" in b.lower() for b in o.bearish)


def test_trend_up_bullish_down_bearish():
    up = build_outlook(_td(), _c(trend="up"), _s())
    dn = build_outlook(_td(), _c(trend="down"), _s())
    assert any("uptrend" in b.lower() for b in up.bullish)
    assert any("downtrend" in b.lower() for b in dn.bearish)
    assert not dn.bullish[0].startswith("Uptrend")


def test_candle_bias_maps_to_side():
    bu = build_outlook(_td(), _c(bias="bullish"), _s())
    be = build_outlook(_td(), _c(bias="bearish"), _s())
    assert any("candle" in b.lower() for b in bu.bullish)
    assert any("candle" in b.lower() for b in be.bearish)


def test_no_signals_yields_neutral_line_each_side():
    o = build_outlook(_td(), _c(), _s())
    assert o.bullish == ["No strong bullish signals"]
    assert o.bearish == ["No strong bearish signals"]
    assert isinstance(o, Outlook) and o.narrative is None


def test_high_volatility_bearish():
    o = build_outlook(_td(), _c(), _s(risk=14.2))
    assert any("14.2" in b for b in o.bearish)


def test_52w_proximity_from_fundamentals_and_prices():
    f = {"fiftyTwoWeekLow": 98.0, "fiftyTwoWeekHigh": 300.0}
    assert any("52-week low" in b for b in build_outlook(_td(fundamentals=f), _c(), _s()).bullish)
    f = {"fiftyTwoWeekLow": 10.0, "fiftyTwoWeekHigh": 102.0}
    assert any("52-week high" in b for b in build_outlook(_td(fundamentals=f), _c(), _s()).bearish)
    c = np.linspace(200, 100, 260)  # ends at its 252-row low
    o = build_outlook(_td("equity", prices=_prices(c)), _c(), _s())
    assert any("52-week low" in b for b in o.bullish)


def test_low_bullet_only_for_equity_and_etf():
    f = {"fiftyTwoWeekLow": 98.0, "fiftyTwoWeekHigh": 300.0}
    for ac in ("equity", "etf"):
        assert any("52-week low" in b for b in build_outlook(_td(ac, f), _c(), _s()).bullish)
    for ac in ("crypto", "commodity"):
        assert not any("52-week" in b for b in build_outlook(_td(ac, f), _c(), _s()).bullish)
    f = {"fiftyTwoWeekLow": 10.0, "fiftyTwoWeekHigh": 102.0}
    for ac in ("crypto", "commodity"):
        assert any("52-week high" in b for b in build_outlook(_td(ac, f), _c(), _s()).bearish)


def test_short_series_gives_no_price_derived_52w():
    c = np.linspace(200, 100, 100)  # at its own range low, but only 100 bars
    o = build_outlook(_td(prices=_prices(c)), _c(), _s())
    assert not any("52-week" in b for b in o.bullish + o.bearish)
    c = np.linspace(100, 200, 40)  # < 60 bars, at its high
    o = build_outlook(_td(prices=_prices(c)), _c(), _s())
    assert not any("52-week" in b for b in o.bullish + o.bearish)


def test_narrow_range_emits_only_nearer_extreme():
    f = {"fiftyTwoWeekLow": 100 / 1.02, "fiftyTwoWeekHigh": 100 / 0.98}
    both = build_outlook(_td(fundamentals=f), _c(), _s())  # price 100: tie
    assert not any("52-week" in b for b in both.bullish + both.bearish)
    f = {"fiftyTwoWeekLow": 99.0, "fiftyTwoWeekHigh": 103.0}  # nearer the low
    o = build_outlook(_td(fundamentals=f), _c(), _s())
    assert any("52-week low" in b for b in o.bullish)
    assert not any("52-week" in b for b in o.bearish)


def test_price_beyond_stale_extreme_is_stated_honestly():
    f = {"fiftyTwoWeekLow": 110.0, "fiftyTwoWeekHigh": 300.0}
    o = build_outlook(_td(fundamentals=f), _c(), _s())
    assert any("at/below its 52-week low 110.00" in b for b in o.bullish)
    f = {"fiftyTwoWeekLow": 10.0, "fiftyTwoWeekHigh": 90.0}
    o = build_outlook(_td(fundamentals=f), _c(), _s())
    assert any("at/above its 52-week high 90.00" in b for b in o.bearish)


def test_sources_not_mixed():
    c = np.linspace(200, 100, 260)  # price-derived would flag the low
    o = build_outlook(_td(fundamentals={"fiftyTwoWeekLow": 50.0}, prices=_prices(c)), _c(), _s())
    # fundamentals incomplete -> both from prices (low 100, high 200) -> low bullet
    assert any("52-week low 100.00" in b for b in o.bullish)


def test_boundaries_are_exclusive_or_inclusive_as_specified():
    for rsi in (30.0, 70.0):
        o = build_outlook(_td(), _c(rsi=rsi), _s())
        assert o.bullish == ["No strong bullish signals"] and o.bearish == ["No strong bearish signals"]
    o = build_outlook(_td(), _c(), _s(risk=10.0))
    assert o.bearish == ["No strong bearish signals"]
    o = build_outlook(_td(fundamentals={"trailingPE": 20.0}), _c(), _s(), sector_median_pe=20.0)
    assert o.bullish == ["No strong bullish signals"] and o.bearish == ["No strong bearish signals"]
    f = {"fiftyTwoWeekLow": 100.0 / 1.05, "fiftyTwoWeekHigh": 400.0}  # exactly 5% above low
    assert any("52-week low" in b for b in build_outlook(_td(fundamentals=f), _c(), _s()).bullish)
    f = {"fiftyTwoWeekLow": 10.0, "fiftyTwoWeekHigh": 100.0 / 0.95}  # exactly 5% below high
    assert any("52-week high" in b for b in build_outlook(_td(fundamentals=f), _c(), _s()).bearish)


def test_pe_bullets_equity_only():
    f = {"trailingPE": 10.0}
    o = build_outlook(_td(fundamentals=f), _c(), _s(), sector_median_pe=20.0)
    assert any("P/E" in b and "10.0" in b and "20.0" in b for b in o.bullish)
    o = build_outlook(_td(fundamentals={"trailingPE": 30.0}), _c(), _s(), sector_median_pe=20.0)
    assert any("P/E" in b for b in o.bearish)
    o = build_outlook(_td(fundamentals=f), _c(), _s())  # no median -> skipped
    assert o.bullish == ["No strong bullish signals"]


def test_commodity_skips_pe_bullets():
    o = build_outlook(_td("commodity", {"trailingPE": 5.0}), _c(), _s(), sector_median_pe=20.0)
    assert not any("P/E" in b for b in o.bullish + o.bearish)


def test_insufficient_data_is_stated_plainly():
    c = _c(rsi=None, note="insufficient data: need 30 bars")
    s = StopSuggestion("TEST", None, None, None, None, "n/a", "insufficient data: need 60 bars")
    o = build_outlook(_td(prices=_prices([100.0] * 5)), c, s)
    text = " ".join(o.bullish + o.bearish).lower()
    assert "insufficient data" in text
    assert "technical" in text and "stop" in text


def test_nan_inputs_do_not_crash():
    o = build_outlook(_td(fundamentals={"trailingPE": float("nan")}),
                      _c(rsi=float("nan")), _s(risk=float("nan")), sector_median_pe=float("nan"))
    assert o.bullish and o.bearish


class _Msg:
    def __init__(self, content, stop_reason=None):
        self.content = content
        if stop_reason is not None:
            self.stop_reason = stop_reason


class _Block:
    def __init__(self, text):
        self.text = text


class _Stub:
    def __init__(self, content=None, exc=None, stop_reason=None):
        self.calls = []
        self._content, self._exc, self._stop = content, exc, stop_reason
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self._exc:
            raise self._exc
        return _Msg(self._content, self._stop)


def test_narrate_passes_only_supplied_facts():
    o = Outlook("ZZZZ", ["secret bullet 9876.5"], ["secret bear 4321.0"], "old ZZZZ")
    stub = _Stub([_Block("ok")])
    narrate(o, {"ticker": "AAA", "rsi": 24.3}, stub)
    kw = stub.calls[0]
    blob = str(kw["messages"])
    assert "24.3" in blob and "AAA" in blob
    for leak in ("ZZZZ", "9876.5", "4321.0", "secret"):
        assert leak not in blob
    assert "not financial advice" in kw["system"].lower()
    assert "120" in kw["system"] and "figures" in kw["system"].lower()
    assert kw["model"] == "claude-sonnet-5-5"


def test_narrate_returns_text():
    o = Outlook("AAA", [], [], None)
    assert narrate(o, {"x": 1}, _Stub([_Block("hello "), _Block("world")])) == "hello world"


def test_narrate_wraps_api_error_and_empty_text():
    o = Outlook("AAA", [], [], None)
    with pytest.raises(NarrationError, match="boom"):
        narrate(o, {}, _Stub(exc=RuntimeError("boom")))
    for content in ([], None, [_Block("  ")], [object()]):
        with pytest.raises(NarrationError, match="no text"):
            narrate(o, {}, _Stub(content))


def test_narrate_max_tokens_is_generous():
    from portfolio_analyzer.outlook import MAX_TOKENS

    o = Outlook("AAA", [], [], None)
    stub = _Stub([_Block("ok")])
    narrate(o, {}, stub)
    assert MAX_TOKENS == 1500 and stub.calls[0]["max_tokens"] == 1500
    assert set(stub.calls[0]) == {"model", "max_tokens", "system", "messages"}


@pytest.mark.parametrize("reason", ["max_tokens", "refusal"])
def test_narrate_bad_stop_reason_is_error_even_with_text(reason):
    o = Outlook("AAA", [], [], None)
    with pytest.raises(NarrationError, match=reason):
        narrate(o, {}, _Stub([_Block("partial text")], stop_reason=reason))


def test_narrate_end_turn_with_text_ignoring_non_text_blocks():
    o = Outlook("AAA", [], [], None)
    thinking = object()
    stub = _Stub([thinking, _Block("fine")], stop_reason="end_turn")
    assert narrate(o, {}, stub) == "fine"
