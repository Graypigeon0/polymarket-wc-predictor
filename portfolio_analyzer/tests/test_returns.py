import numpy as np
import pandas as pd

from portfolio_analyzer.returns import positive_price_returns


def _s(vals):
    return pd.Series(vals, index=pd.bdate_range("2024-01-01", periods=len(vals)), dtype=float)


def test_negative_price_returns_excluded_and_counted():
    rets, n = positive_price_returns(_s([10, -5, 8, 9]))
    assert n == 2
    assert len(rets) == 1
    assert abs(rets.iloc[0] - (9 / 8 - 1)) < 1e-12


def test_zero_price_excluded():
    rets, n = positive_price_returns(_s([10, 0, 5, 6]))
    assert n == 2
    assert len(rets) == 1


def test_all_positive_identical_to_pct_change():
    s = _s([10, 11, 9.5, 12, 12.5])
    rets, n = positive_price_returns(s)
    assert n == 0
    pd.testing.assert_series_equal(rets, s.pct_change(fill_method=None).dropna())


def test_dataframe_masks_per_column_and_counts():
    idx = pd.bdate_range("2024-01-01", periods=4)
    df = pd.DataFrame({"A": [10, -5, 8, 9], "B": [1, 2, 3, 4]}, index=idx, dtype=float)
    rets, n = positive_price_returns(df)
    assert n["A"] == 2 and n["B"] == 0
    assert rets["A"].notna().sum() == 1
    assert np.isclose(rets["B"].iloc[2], 0.5)
