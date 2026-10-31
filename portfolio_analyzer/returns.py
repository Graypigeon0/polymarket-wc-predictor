"""Daily simple returns that ignore moves across non-positive prices. Pure; no network."""

from __future__ import annotations

import numpy as np
import pandas as pd


def positive_price_returns(
    close: pd.Series | pd.DataFrame,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | int]:
    """Daily simple returns (NaN/inf dropped) and how many were excluded for bad prices.

    A return is masked when its previous OR current close is <= 0, since a percentage change
    is meaningless there (e.g. crude oil 18.27 -> -37.63 -> 10.01). Uses
    pct_change(fill_method=None). For a Series, returns (returns, excluded_count). For a
    DataFrame, returns the returns frame with NaN left in place for masked/missing cells
    (callers drop rows) and a per-column Series of excluded counts.
    """
    close = close.astype(float)
    bad = (close <= 0) | (close.shift(1) <= 0)
    rets = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    excluded = (bad & close.notna() & close.shift(1).notna()).sum()
    rets = rets.mask(bad)
    if isinstance(close, pd.Series):
        return rets.dropna(), int(excluded)
    return rets, excluded.astype(int)
