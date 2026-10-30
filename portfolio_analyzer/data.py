"""yfinance data layer with a per-ticker, per-day on-disk cache.

This is the only module in the package allowed to touch the network.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from yfinance import exceptions as yf_exc

from portfolio_analyzer.models import AssetClass, TickerData

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
FUNDAMENTAL_KEYS = (
    "trailingPE",
    "pegRatio",
    "freeCashflow",
    "marketCap",
    "sector",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
)
_QUOTE_TYPES: dict[str, AssetClass] = {
    "CRYPTOCURRENCY": "crypto",
    "FUTURE": "commodity",
    "ETF": "etf",
    "EQUITY": "equity",
}


class DataError(Exception):
    """Raised when market data is empty, invalid or unavailable."""


class DataProviderError(DataError):
    """Raised when the data provider (Yahoo Finance) cannot be reached or refuses the request."""


def classify(quote_type: str, ticker: str) -> AssetClass:
    known = _QUOTE_TYPES.get((quote_type or "").upper())
    if known:
        return known
    t = ticker.upper()
    if t.endswith("=F"):
        return "commodity"
    if t.endswith("-USD"):
        return "crypto"
    return "equity"


def annualisation_factor(asset_class: AssetClass) -> int:
    return 365 if asset_class == "crypto" else 252


def _safe_name(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)


def _read_cache(prices_path: Path, meta_path: Path) -> TickerData | None:
    if not (prices_path.exists() and meta_path.exists()):
        return None
    try:
        prices = pd.read_pickle(prices_path)
        meta = json.loads(meta_path.read_text())
        return TickerData(meta["ticker"], meta["asset_class"], prices, meta["fundamentals"])
    except Exception:
        return None  # corrupt cache: refetch


def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise DataError(f"No price data returned for {ticker!r}")
    missing = [c for c in PRICE_COLUMNS if c not in raw.columns]
    if missing:
        raise DataError(f"Price data for {ticker!r} lacks columns {missing}")
    df = raw[PRICE_COLUMNS].copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.empty:
        raise DataError(f"No usable price rows for {ticker!r}")
    return df


def fetch_ticker(
    ticker: str,
    start: str = "2019-12-01",
    cache_dir: Path = Path(".cache/portfolio_analyzer"),
) -> TickerData:
    cache_dir = Path(cache_dir)
    stem = f"{_safe_name(ticker)}_{start}_{date.today().isoformat()}"
    prices_path = cache_dir / f"{stem}.pkl"
    meta_path = cache_dir / f"{stem}.json"
    cached = _read_cache(prices_path, meta_path)
    if cached is not None:
        return cached

    try:
        tk = yf.Ticker(ticker)
        # raise_errors=True: by default yfinance swallows network failures and returns an
        # empty frame, indistinguishable from a delisted symbol.
        raw = tk.history(start=start, auto_adjust=True, raise_errors=True)
    except (yf_exc.YFTickerMissingError, yf_exc.YFPricesMissingError, ValueError) as exc:
        raise DataError(f"No price data returned for {ticker!r}: {exc}") from exc
    except Exception as exc:
        raise DataProviderError(
            f"Could not reach Yahoo Finance while fetching {ticker!r}: {type(exc).__name__}: {exc}"
        ) from exc
    prices = _normalise(raw, ticker)

    # Quote type: the chart metadata needs no crumb; Ticker.info may be 401/rate-limited.
    quote_type = ""
    try:
        quote_type = str(tk.history_metadata.get("instrumentType") or "")
    except Exception:
        pass
    info: dict[str, Any] = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    quote_type = quote_type or str(info.get("quoteType") or "")
    fundamentals = {k: info[k] for k in FUNDAMENTAL_KEYS if info.get(k) is not None}

    asset_class = classify(quote_type, ticker)
    result = TickerData(ticker, asset_class, prices, fundamentals)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        prices.to_pickle(prices_path)
        meta_path.write_text(
            json.dumps(
                {"ticker": ticker, "asset_class": asset_class, "fundamentals": fundamentals},
                default=str,
            )
        )
    except OSError:
        pass  # cache is best-effort
    return result
