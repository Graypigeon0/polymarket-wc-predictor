from __future__ import annotations

from portfolio_analyzer.models import Holding


def parse_holdings(args: list[str]) -> tuple[list[Holding], list[str]]:
    """Parse ``TICKER[:WEIGHT]`` args into holdings whose weights sum to 1.0.

    Weights are percentages. Either all or none of the args carry a weight.
    """
    if not args:
        raise ValueError("No holdings given")

    tickers: list[str] = []
    raw_weights: list[float | None] = []
    for arg in args:
        arg = arg.strip()
        ticker, sep, weight_str = arg.rpartition(":")
        if not sep:
            ticker, weight_str = arg, ""
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError(f"Invalid holding {arg!r}: missing ticker")
        weight: float | None = None
        if sep:
            try:
                weight = float(weight_str)
            except ValueError:
                raise ValueError(f"Invalid weight for {ticker}: {weight_str!r}") from None
            if not weight > 0 or weight == float("inf"):
                raise ValueError(f"Weight for {ticker} must be positive, got {weight_str!r}")
        if ticker in tickers:
            raise ValueError(f"Duplicate ticker: {ticker}")
        tickers.append(ticker)
        raw_weights.append(weight)

    warnings: list[str] = []
    provided = [w is not None for w in raw_weights]
    if all(provided):
        total = sum(w for w in raw_weights if w is not None)
        if abs(total - 100.0) > 1e-9:
            warnings.append(f"Weights sum to {total:g}%, normalised to 100%")
        weights = [w / total for w in raw_weights if w is not None]
    elif not any(provided):
        weights = [1.0 / len(tickers)] * len(tickers)
    else:
        raise ValueError("Specify weights for all or none of the holdings")

    return [Holding(t, w) for t, w in zip(tickers, weights, strict=True)], warnings
