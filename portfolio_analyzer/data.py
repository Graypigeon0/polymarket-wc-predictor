"""yfinance data layer with a per-ticker, per-day on-disk cache.

This is the only module in the package allowed to touch the network.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
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


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "portfolio_analyzer"


def _now() -> datetime:
    """Current time, timezone-aware UTC. Module-level so tests can inject a clock."""
    return datetime.now(UTC)


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass  # best-effort: the cache is optional


def _read_cache(prices_path: Path, meta_path: Path) -> tuple[TickerData, dict[str, Any]] | None:
    """Cached data plus its meta, or None when missing, corrupt or in an old format."""
    if not (prices_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        info_ok, resolved = meta["info_ok"], meta["type_resolved"]
        if not (isinstance(info_ok, bool) and isinstance(resolved, bool)):
            return None
        prices = pd.read_csv(
            prices_path, index_col=0, parse_dates=True, float_precision="round_trip"
        )
        prices.index = pd.DatetimeIndex(prices.index, name=None)
        prices = prices[PRICE_COLUMNS].astype(float)
        if prices.empty:
            return None
        td = TickerData(meta["ticker"], meta["asset_class"], prices, dict(meta["fundamentals"]))
        return td, meta
    except Exception:
        return None  # corrupt or old-format cache: refetch


def _write_cache(
    cache_dir: Path, prices_path: Path, meta_path: Path, td: TickerData, flags: dict[str, bool]
) -> None:
    try:
        _ensure_private_dir(cache_dir)
        td.prices.to_csv(prices_path)
        meta = {
            "ticker": td.ticker,
            "asset_class": td.asset_class,
            "fundamentals": td.fundamentals,
            **flags,
        }
        meta_path.write_text(json.dumps(meta, default=str))
    except OSError:
        pass  # cache is best-effort


def _drop_unfinished_bar(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop the final row when it is dated today in the instrument's exchange time zone.

    During market hours (and always for 24h crypto) yfinance returns today's in-progress bar;
    treating it as a complete daily candle skews patterns, ATR, RSI and stops. Removing it
    keeps the per-day cache valid for the whole day.
    """
    if raw is None or raw.empty:
        return raw
    idx = pd.DatetimeIndex(raw.index)
    now = _now()
    today = now.astimezone(idx.tz).date() if idx.tz is not None else now.date()
    if idx[-1].date() == today:
        return raw.iloc[:-1]
    return raw


def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise DataError(f"No price data returned for {ticker!r}")
    missing = [c for c in PRICE_COLUMNS if c not in raw.columns]
    if missing:
        raise DataError(f"Price data for {ticker!r} lacks columns {missing}")
    df = raw[PRICE_COLUMNS].astype(float)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx.normalize(), freq=None, name=None)
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.empty:
        raise DataError(f"No usable price rows for {ticker!r}")
    return df


def _lookup_quote_type(tk: Any) -> str:
    """Quote type from the chart metadata (needs no crumb); '' when unavailable."""
    try:
        return str(tk.history_metadata.get("instrumentType") or "")
    except Exception:
        return ""


def _lookup_info(tk: Any) -> dict[str, Any] | None:
    """Ticker.info (may be 401/429); None on failure."""
    try:
        return dict(tk.info or {}) or None
    except Exception:
        return None


def _fundamentals(info: dict[str, Any]) -> dict[str, Any]:
    return {k: info[k] for k in FUNDAMENTAL_KEYS if info.get(k) is not None}


def fetch_ticker(
    ticker: str,
    start: str = "2019-12-01",
    cache_dir: Path | None = None,
) -> TickerData:
    cache_dir = default_cache_dir() if cache_dir is None else Path(cache_dir)
    stem = f"{_safe_name(ticker)}_{start}_{date.today().isoformat()}"
    prices_path = cache_dir / f"{stem}.csv"
    meta_path = cache_dir / f"{stem}.json"
    cached = _read_cache(prices_path, meta_path)
    if cached is not None:
        td, meta = cached
        if meta["info_ok"] and meta["type_resolved"]:
            return td
        return _refresh_lookups(ticker, td, meta, cache_dir, prices_path, meta_path)

    try:
        tk = yf.Ticker(ticker)
        # raise_errors=True: by default yfinance swallows network failures and returns an
        # empty frame, indistinguishable from a delisted symbol.
        raw = tk.history(start=start, auto_adjust=True, raise_errors=True)
    except (yf_exc.YFTickerMissingError, yf_exc.YFPricesMissingError, ValueError) as exc:
        raise DataError(f"No price data returned for {ticker!r}: {exc}") from exc
    except TypeError as exc:
        # A signature mismatch means the yfinance API changed; it is not a Yahoo outage.
        raise RuntimeError(f"yfinance API changed: {exc}") from exc
    except Exception as exc:
        raise DataProviderError(
            f"Could not reach Yahoo Finance while fetching {ticker!r}: {type(exc).__name__}: {exc}"
        ) from exc
    prices = _normalise(_drop_unfinished_bar(raw), ticker)

    quote_type = _lookup_quote_type(tk)
    info = _lookup_info(tk)
    quote_type = quote_type or str((info or {}).get("quoteType") or "")
    result = TickerData(
        ticker, classify(quote_type, ticker), prices, _fundamentals(info or {})
    )
    flags = {"info_ok": info is not None, "type_resolved": bool(quote_type)}
    _write_cache(cache_dir, prices_path, meta_path, result, flags)
    return result


def _refresh_lookups(
    ticker: str,
    td: TickerData,
    meta: dict[str, Any],
    cache_dir: Path,
    prices_path: Path,
    meta_path: Path,
) -> TickerData:
    """Retry (once) only the lookups that failed earlier; update the cache on success only."""
    tk = yf.Ticker(ticker)
    info_ok, resolved = bool(meta["info_ok"]), bool(meta["type_resolved"])
    asset_class, fundamentals = td.asset_class, td.fundamentals
    quote_type = "" if resolved else _lookup_quote_type(tk)
    if not info_ok:
        info = _lookup_info(tk)
        if info is not None:
            info_ok, fundamentals = True, _fundamentals(info)
            quote_type = quote_type or str(info.get("quoteType") or "")
    if quote_type:
        resolved, asset_class = True, classify(quote_type, ticker)
    if info_ok == meta["info_ok"] and resolved == meta["type_resolved"]:
        return td  # nothing improved; leave the cache as it was
    updated = TickerData(td.ticker, asset_class, td.prices, fundamentals)
    _write_cache(
        cache_dir, prices_path, meta_path, updated, {"info_ok": info_ok, "type_resolved": resolved}
    )
    return updated
