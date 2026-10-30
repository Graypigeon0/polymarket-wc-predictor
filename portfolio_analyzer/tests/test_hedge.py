import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer.hedge import analyse_hedge
from portfolio_analyzer.models import Holding, TickerData


def _td(ticker, close, asset_class="equity", sector=None):
    close = close.astype(float)
    df = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}
    )
    return TickerData(ticker, asset_class, df, {"sector": sector} if sector else {})


def _rets(n=400, seed=0, start="2019-06-03"):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.01, n), index=idx)


def _from_rets(r):
    return 100 * (1 + r).cumprod()


def test_beta_of_benchmark_clone_is_one():
    r = _rets()
    bench = _td("SPY", _from_rets(r), "etf")
    res = analyse_hedge([Holding("SPY", 1.0)], {"SPY": bench}, bench)
    assert abs(res.beta - 1.0) < 1e-9
    assert res.high_corr_pairs == []
    assert res.correlation.shape == (1, 1)


def test_single_holding_benchmark_clone_verdict_none():
    r = _rets()
    bench = _td("SPY", _from_rets(r), "etf")
    res = analyse_hedge([Holding("SPY", 1.0)], {"SPY": bench}, bench)
    assert res.verdict == "none"


def test_beta_of_half_cash_like_asset_is_about_half():
    r = _rets()
    bench = _td("SPY", _from_rets(r), "etf")
    half = _td("HALF", _from_rets(r * 0.5), "etf")
    res = analyse_hedge([Holding("HALF", 1.0)], {"HALF": half}, bench)
    assert abs(res.beta - 0.5) < 0.01


def test_negatively_correlated_holding_listed_as_offset():
    r = _rets()
    bench = _td("SPY", _from_rets(r), "etf")
    inv = _td("INV", _from_rets(-r * 0.8), "etf")
    other = _td("AAA", _from_rets(r + _rets(seed=5) * 0.5))
    other2 = _td("BBB", _from_rets(r + _rets(seed=6) * 0.5))
    res = analyse_hedge(
        [Holding("INV", 0.2), Holding("AAA", 0.4), Holding("BBB", 0.4)],
        {"INV": inv, "AAA": other, "BBB": other2},
        bench,
    )
    assert "INV" in res.offsets
    assert "AAA" not in res.offsets


def test_high_corr_pair_flagged():
    r = _rets()
    bench = _td("SPY", _from_rets(_rets(seed=9)), "etf")
    a = _td("A", _from_rets(r))
    b = _td("B", _from_rets(r * 1.1))
    c = _td("C", _from_rets(_rets(seed=3)))
    res = analyse_hedge(
        [Holding("A", 0.4), Holding("B", 0.3), Holding("C", 0.3)],
        {"A": a, "B": b, "C": c},
        bench,
    )
    pairs = {(x, y) for x, y, _ in res.high_corr_pairs}
    assert ("A", "B") in pairs
    assert not any("C" in p for p in pairs)


def test_concentration_flags_over_50_percent():
    r = _rets()
    bench = _td("SPY", _from_rets(r), "etf")
    a = _td("A", _from_rets(_rets(seed=1)), sector="Tech")
    b = _td("B", _from_rets(_rets(seed=2)), sector="Tech")
    c = _td("C", _from_rets(_rets(seed=3)), sector="Energy")
    res = analyse_hedge(
        [Holding("A", 0.4), Holding("B", 0.3), Holding("C", 0.3)],
        {"A": a, "B": b, "C": c},
        bench,
    )
    assert abs(res.concentration["Tech"] - 0.7) < 1e-9
    assert abs(res.concentration["Energy"] - 0.3) < 1e-9
    assert "Tech" in res.reason and "50%" in res.reason


def test_stress_window_skipped_when_history_short():
    r = _rets(n=300, start="2023-01-02")
    bench = _td("SPY", _from_rets(r), "etf")
    res = analyse_hedge([Holding("SPY", 1.0)], {"SPY": bench}, bench)
    assert [s.name for s in res.stress] == ["covid_2020", "bear_2022"]
    for s in res.stress:
        assert s.portfolio_return is None and s.benchmark_return is None
        assert s.skipped_reason


def test_stress_return_computed_when_covered():
    r = _rets(n=1000, start="2019-06-03")
    bench = _td("SPY", _from_rets(r), "etf")
    h = _td("H", _from_rets(r * 0.5), "etf")
    res = analyse_hedge([Holding("H", 1.0)], {"H": h}, bench)
    covid = res.stress[0]
    assert covid.skipped_reason is None
    idx = r.index
    w = r[(idx > "2020-02-19") & (idx <= "2020-03-23")]  # close(start) -> close(end)
    assert abs(covid.benchmark_return - ((1 + w).prod() - 1)) < 1e-9
    assert abs(covid.portfolio_return - ((1 + w * 0.5).prod() - 1)) < 1e-9


def _weekend_btc_and_bench(weeks=100, seed=4):
    """BTC moves only on Sat/Sun; the weekday benchmark's Monday return tracks that move."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-03", periods=weeks * 7, freq="D")  # starts on a Monday
    btc_r = pd.Series(0.0, index=days)
    wknd = days[days.weekday >= 5]
    btc_r[wknd] = rng.normal(0, 0.03, len(wknd))
    btc_close = 100 * (1 + btc_r).cumprod()
    bdays = days[days.weekday < 5]
    bench_r = pd.Series(rng.normal(0, 0.0005, len(bdays)), index=bdays)
    for monday in bdays[bdays.weekday == 0][1:]:
        weekend = btc_close[monday] / btc_close[monday - pd.Timedelta(days=3)] - 1
        bench_r[monday] = 0.5 * weekend
    bench_close = 100 * (1 + bench_r).cumprod()
    return _td("BTC-USD", btc_close, "crypto"), _td("SPY", bench_close, "etf")


def test_crypto_weekend_moves_compounded_into_monday_return():
    from portfolio_analyzer.hedge import joined_returns

    btc, spy = _weekend_btc_and_bench()
    j = joined_returns([Holding("BTC-USD", 1.0)], {"BTC-USD": btc}, spy)
    assert not j.isna().any().any()
    assert all(d.weekday() < 5 for d in j.index)
    close = btc.prices["Close"]
    monday = pd.Timestamp("2022-01-17")
    friday = pd.Timestamp("2022-01-14")
    expected = close[monday] / close[friday] - 1  # Fri->Mon = Sat * Sun * Mon moves
    assert abs(expected) > 1e-3
    assert abs(j.loc[monday, "BTC-USD"] - expected) < 1e-12
    # a weekday-only return is NOT the weekend-less Sun->Mon move
    sun_mon = close[monday] / close[monday - pd.Timedelta(days=1)] - 1
    assert abs(j.loc[monday, "BTC-USD"] - sun_mon) > 1e-4


def test_weekend_only_crypto_has_nonzero_beta():
    btc, spy = _weekend_btc_and_bench()
    res = analyse_hedge([Holding("BTC-USD", 1.0)], {"BTC-USD": btc}, spy)
    assert res.stress != []
    assert abs(res.beta) > 0.5


def test_holiday_mismatch_uses_common_dates_price_change():
    from portfolio_analyzer.hedge import joined_returns

    idx = pd.bdate_range("2023-01-02", periods=80)
    rng = np.random.default_rng(1)
    bench_close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 80)), index=idx)
    h_close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 80)), index=idx)
    bench = _td("SPY", bench_close.drop(idx[10]), "etf")  # benchmark market closed that day
    h = _td("H", h_close, "etf")
    j = joined_returns([Holding("H", 1.0)], {"H": h}, bench)
    assert idx[10] not in j.index
    assert abs(j.loc[idx[11], "H"] - (h_close[idx[11]] / h_close[idx[9]] - 1)) < 1e-12


def test_59_observations_is_unavailable_and_names_limiting_ticker():
    # 61 prices -> 60 returns for the benchmark; NEW-USD starts later giving only 59 joined
    r = _rets(n=400, start="2023-01-02")
    bench = _td("SPY", _from_rets(r), "etf")
    tail = _from_rets(_rets(n=60, seed=3, start="2024-01-01"))  # 60 prices -> 59 returns
    new = _td("NEW-USD", tail, "etf")
    res = analyse_hedge([Holding("NEW-USD", 1.0)], {"NEW-USD": new}, bench)
    assert res.stress == []
    assert "NEW-USD" in res.reason
    assert str(tail.index.min().date()) in res.reason
    assert "59" in res.reason


def test_60_observations_is_available():
    r = _rets(n=400, start="2023-01-02")
    bench = _td("SPY", _from_rets(r), "etf")
    tail = _from_rets(_rets(n=61, seed=3, start="2024-01-01"))  # 61 prices -> 60 returns
    new = _td("NEW-USD", tail, "etf")
    res = analyse_hedge([Holding("NEW-USD", 1.0)], {"NEW-USD": new}, bench)
    assert res.stress != []
    assert "Based on 60 daily observations" in res.reason


def test_reason_includes_observation_count_and_date_range():
    r = _rets(n=300, start="2023-01-02")
    bench = _td("SPY", _from_rets(r), "etf")
    h = _td("H", _from_rets(r * 0.9), "etf")
    res = analyse_hedge([Holding("H", 1.0)], {"H": h}, bench)
    first, last = r.index[1].date(), r.index[-1].date()
    assert f"Based on 299 daily observations, {first} to {last}." in res.reason


def test_single_holding_never_listed_as_its_own_offset():
    r = _rets()
    bench = _td("SPY", _from_rets(r), "etf")
    low = _td("LOW", _from_rets(r * 0.1 + _rets(seed=8) * 0.5), "etf")
    res = analyse_hedge([Holding("LOW", 1.0)], {"LOW": low}, bench)
    assert res.offsets == []


def test_verdict_effective():
    r = _rets(n=1000)
    for lo, hi in (("2020-02-19", "2020-03-23"), ("2022-01-03", "2022-10-12")):
        r.loc[lo:hi] -= 0.01  # market falls through both windows
    bench = _td("SPY", _from_rets(r), "etf")
    h = _td("H", _from_rets(r * 0.4), "etf")
    res = analyse_hedge([Holding("H", 1.0)], {"H": h}, bench)
    assert res.beta <= 0.7
    assert all(s.portfolio_return > s.benchmark_return for s in res.stress)
    assert res.verdict == "effective"


def test_verdict_partial():
    r = _rets(n=1000)
    bench = _td("SPY", _from_rets(r), "etf")
    h = _td("H", _from_rets(r * 0.9), "etf")
    res = analyse_hedge([Holding("H", 1.0)], {"H": h}, bench)
    assert 0.7 < res.beta < 1.0
    assert res.verdict == "partial"
    assert "beta" in res.reason


def test_insufficient_overlap_is_safe():
    r = _rets(n=1)
    bench = _td("SPY", _from_rets(r), "etf")
    res = analyse_hedge([Holding("SPY", 1.0)], {"SPY": bench}, bench)
    assert res.verdict == "none"
    assert "insufficient overlapping history" in res.reason.lower()


@pytest.mark.parametrize("seed", range(20))
def test_benchmark_clone_verdict_none_across_seeds(seed):
    r = _rets(seed=seed)
    bench = _td("SPY", _from_rets(r), "etf")
    res = analyse_hedge([Holding("SPY", 1.0)], {"SPY": bench}, bench)
    assert res.beta == 1.0 and res.verdict == "none"
    clone2 = _td("QQQ", _from_rets(r), "etf")
    res = analyse_hedge(
        [Holding("SPY", 0.3), Holding("QQQ", 0.7)], {"SPY": bench, "QQQ": clone2}, bench
    )
    assert res.beta == 1.0 and res.verdict == "none"


def test_missing_holding_renormalised_and_named():
    r = _rets()
    bench = _td("SPY", _from_rets(r), "etf")
    res = analyse_hedge([Holding("SPY", 0.6), Holding("MISSING", 0.4)], {"SPY": bench}, bench)
    assert res.beta == 1.0
    assert res.verdict == "none"
    assert "MISSING" in res.reason and "no data" in res.reason
    assert abs(sum(res.concentration.values()) - 1.0) < 1e-9


def test_all_holdings_missing_is_safe():
    bench = _td("SPY", _from_rets(_rets()), "etf")
    res = analyse_hedge([Holding("X", 1.0)], {}, bench)
    assert res.verdict == "none"
    assert "X" in res.reason


def test_zero_variance_benchmark_is_safe():
    idx = pd.bdate_range("2023-01-02", periods=100)
    flat = _td("FLAT", pd.Series(100.0, index=idx), "etf")
    h = _td("H", _from_rets(_rets(n=100, start="2023-01-02")), "etf")
    res = analyse_hedge([Holding("H", 1.0)], {"H": h}, flat)
    assert res.verdict == "none"
    assert res.offsets == []
    assert "benchmark has no variance over the overlapping history" in res.reason.lower()


def test_low_beta_no_covered_window_is_partial_with_withheld_note():
    r = _rets(n=300, start="2023-01-02")
    bench = _td("SPY", _from_rets(r), "etf")
    h = _td("H", _from_rets(r * 0.4), "etf")
    res = analyse_hedge([Holding("H", 1.0)], {"H": h}, bench)
    assert res.verdict == "partial"
    assert "withheld" in res.reason and "no stress window could be tested" in res.reason


def test_skipped_reason_names_latest_starting_ticker():
    r = _rets(n=1000)
    bench = _td("SPY", _from_rets(r), "etf")
    short = _td("NEW", _from_rets(r.iloc[-300:] * 0.5), "etf")
    res = analyse_hedge([Holding("NEW", 1.0)], {"NEW": short}, bench)
    assert all("NEW" in s.skipped_reason for s in res.stress)
    assert str(short.prices.index.min().date()) in res.stress[0].skipped_reason


def test_low_beta_alone_makes_offset():
    r = _rets()
    bench = _td("SPY", _from_rets(r), "etf")
    low = _td("LOW", _from_rets(r * 0.1 + _rets(seed=8) * 0.5), "etf")
    high = _td("HIGH", _from_rets(r * 1.2 + _rets(seed=9) * 0.1), "etf")
    res = analyse_hedge(
        [Holding("LOW", 0.5), Holding("HIGH", 0.5)], {"LOW": low, "HIGH": high}, bench
    )
    assert "LOW" in res.offsets and "HIGH" not in res.offsets


def test_high_beta_losing_windows_is_none():
    r = _rets(n=1000)
    for lo, hi in (("2020-02-19", "2020-03-23"), ("2022-01-03", "2022-10-12")):
        r.loc[lo:hi] -= 0.01
    bench = _td("SPY", _from_rets(r), "etf")
    h = _td("H", _from_rets(r * 1.5), "etf")
    res = analyse_hedge([Holding("H", 1.0)], {"H": h}, bench)
    assert res.beta > 1.0
    assert res.verdict == "none"


def test_mutually_anticorrelated_holdings_both_offsets():
    r = _rets()
    bench = _td("SPY", _from_rets(_rets(seed=11)), "etf")
    a = _td("A", _from_rets(r))
    b = _td("B", _from_rets(-r))
    res = analyse_hedge([Holding("A", 0.5), Holding("B", 0.5)], {"A": a, "B": b}, bench)
    assert set(res.offsets) == {"A", "B"}
