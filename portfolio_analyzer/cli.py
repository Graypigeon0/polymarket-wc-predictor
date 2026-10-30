from __future__ import annotations

import math
import os
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.markdown import Markdown

from portfolio_analyzer import data as data_mod
from portfolio_analyzer.candles import analyse_candles
from portfolio_analyzer.data import DataError, DataProviderError
from portfolio_analyzer.hedge import analyse_hedge
from portfolio_analyzer.holdings import parse_holdings
from portfolio_analyzer.models import HedgeResult, Outlook, TickerData
from portfolio_analyzer.outlook import NarrationError, build_outlook
from portfolio_analyzer.outlook import narrate as narrate_outlook
from portfolio_analyzer.report import HoldingAnalysis, last_close, render_markdown
from portfolio_analyzer.stops import suggest_stop
from portfolio_analyzer.valuation import has_fundamentals, load_universe, screen_underpriced

MIN_SECTOR_PEERS = 3
UNIVERSE_WORKERS = 8
EXIT_USAGE = 2

app = typer.Typer(add_completion=False, help="Portfolio analyzer: decision support, not advice.")
fetch = data_mod.fetch_ticker  # module-level so tests can replace it


def sector_median_pe(universe: dict[str, TickerData], sector: Any) -> float | None:
    """Median positive trailingPE of universe stocks in `sector`; None if <3 peers."""
    if not isinstance(sector, str) or not sector:
        return None
    pes: list[float] = []
    for td in universe.values():
        f = td.fundamentals or {}
        if f.get("sector") != sector:
            continue
        try:
            pe = float(f.get("trailingPE"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(pe) and pe > 0:
            pes.append(pe)
    if len(pes) < MIN_SECTOR_PEERS:
        return None
    return float(statistics.median(pes))


def fetch_universe(
    tickers: list[str],
    fetch_fn: Callable[[str], TickerData],
    on_progress: Callable[[int, int], None] | None = None,
    workers: int = UNIVERSE_WORKERS,
) -> dict[str, TickerData]:
    """Fetch tickers on a thread pool, tolerating per-ticker failures.

    `on_progress(done, total)` is called after each ticker. On KeyboardInterrupt the queued
    fetches are cancelled and the interrupt is re-raised.
    """
    out: dict[str, TickerData] = {}

    def one(t: str) -> TickerData | None:
        try:
            return fetch_fn(t)
        except Exception:  # per-ticker failures are tolerated
            return None

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(one, t): t for t in tickers}
        for done, fut in enumerate(as_completed(futures), start=1):
            td = fut.result()
            if td is not None:
                out[futures[fut]] = td
            if on_progress is not None:
                on_progress(done, len(tickers))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return out


def _fetch_universe(tickers: list[str], warnings: list[str]) -> dict[str, TickerData]:
    if not tickers:
        return {}
    total = len(tickers)
    with Console(stderr=True).status(f"Fetching universe 0/{total}") as status:

        def progress(done: int, total: int) -> None:
            status.update(f"Fetching universe {done}/{total}")

        # look up the module-level `fetch` at call time so tests can replace it
        out = fetch_universe(tickers, lambda t: fetch(t), progress)
    failed = total - len(out)
    if failed:
        warnings.append(f"Universe: {failed} of {total} tickers could not be fetched")
    return out


def _make_client() -> Any:
    """Build the Anthropic client; imports lazily and exits (code 2) if unavailable."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        typer.echo("--narrate needs the ANTHROPIC_API_KEY environment variable.", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        import anthropic
    except ImportError:
        typer.echo("--narrate needs the anthropic package: pip install '.[narrate]'", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    return anthropic.Anthropic()


def _facts(a: HoldingAnalysis, outlook: Outlook) -> dict[str, Any]:
    """Computed figures for this ticker only (no weights, no other tickers)."""
    c, s, d = a.candles, a.stop, a.data
    return {
        "ticker": d.ticker,
        "asset_class": d.asset_class,
        "last_close": last_close(d),
        "trend": c.trend,
        "rsi": c.rsi,
        "candle_bias": c.bias,
        "patterns": [p.name for p in c.patterns],
        "atr_stop": s.atr_stop,
        "support_stop": s.support_stop,
        "recommended_stop": s.recommended,
        "stop_risk_pct": s.risk_pct,
        "trailing_pe": d.fundamentals.get("trailingPE"),
        "sector_median_pe": a.sector_median_pe,
        "bullish": outlook.bullish,
        "bearish": outlook.bearish,
    }


def _unavailable_hedge(reason: str) -> HedgeResult:
    return HedgeResult(0.0, pd.DataFrame(), [], {}, [], [], "none", reason)


@app.command()
def main(
    tickers: Annotated[list[str], typer.Argument(help="TICKER[:WEIGHT] e.g. AAPL:40 BTC-USD:10")],
    benchmark: Annotated[str, typer.Option(help="Benchmark ticker for the hedge.")] = "SPY",
    report: Annotated[Path | None, typer.Option(help="Write the Markdown report here.")] = None,
    narrate: Annotated[bool, typer.Option("--narrate", help="LLM narration (API key).")] = False,
    universe: Annotated[str, typer.Option(help="'sp500' or a file of tickers.")] = "sp500",
) -> None:
    run(tickers, benchmark, report, narrate, universe)


def run(
    tickers: list[str], benchmark: str, report: Path | None, narrate_on: bool, universe: str
) -> None:
    try:
        _run(tickers, benchmark, report, narrate_on, universe)
    except KeyboardInterrupt:
        typer.echo("Interrupted.", err=True)
        raise typer.Exit(130) from None


def _run(
    tickers: list[str], benchmark: str, report: Path | None, narrate_on: bool, universe: str
) -> None:
    console = Console()
    try:
        holdings, warnings = parse_holdings(tickers)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    if universe.strip().lower() != "sp500" and not Path(universe).is_file():
        typer.echo(f"Error: universe file not found: {universe}", err=True)
        raise typer.Exit(EXIT_USAGE)
    client = _make_client() if narrate_on else None
    for w in warnings:
        typer.echo(f"Warning: {w}", err=True)

    fetched: dict[str, TickerData] = {}
    errors: dict[str, str] = {}
    provider_failures = 0
    for h in holdings:
        try:
            fetched[h.ticker] = fetch(h.ticker)
        except DataProviderError as exc:
            errors[h.ticker] = str(exc)
            provider_failures += 1
        except DataError as exc:
            errors[h.ticker] = str(exc)
    if provider_failures == len(holdings):
        typer.echo(
            "Error: the data provider (Yahoo Finance) could not be reached for any holding: "
            + next(iter(errors.values())),
            err=True,
        )
        raise typer.Exit(1)
    if not fetched:
        detail = "; ".join(f"{t}: {m}" for t, m in errors.items())
        typer.echo(f"Error: No holdings could be analysed. {detail}", err=True)
        raise typer.Exit(1)

    bench: TickerData | None = fetched.get(benchmark.upper())
    bench_error = ""
    if bench is None:
        try:
            bench = fetch(benchmark.upper())
        except DataError as exc:
            bench_error = str(exc)

    try:
        names = load_universe(universe)
    except Exception as exc:
        names = []
        warnings.append(f"Universe '{universe}' could not be loaded: {exc}")
    held = {h.ticker for h in holdings}
    universe_data = _fetch_universe([n for n in names if n not in fetched], warnings)
    universe_data.update({t: d for t, d in fetched.items() if t in names})

    analyses: dict[str, HoldingAnalysis] = {}
    skipped: list[str] = []
    for h in holdings:
        td = fetched.get(h.ticker)
        if td is None:
            continue
        candles = analyse_candles(td)
        stop = suggest_stop(td)
        is_equity = td.asset_class == "equity"
        med = sector_median_pe(universe_data, td.fundamentals.get("sector")) if is_equity else None
        outlook = build_outlook(td, candles, stop, med)
        ok = has_fundamentals(td)
        if is_equity and not ok:
            skipped.append(h.ticker)
        a = HoldingAnalysis(td, candles, stop, outlook, med, ok)
        if client is not None:
            try:
                outlook.narrative = narrate_outlook(outlook, _facts(a, outlook), client)
            except NarrationError as exc:
                typer.echo(f"Warning: {exc}", err=True)
                warnings.append(str(exc))
        analyses[h.ticker] = a

    if bench is None:
        hedge = _unavailable_hedge(f"benchmark {benchmark.upper()} unavailable: {bench_error}")
    else:
        hedge = analyse_hedge(holdings, fetched, bench)

    candidates = screen_underpriced({t: d for t, d in universe_data.items() if t not in held})
    md = render_markdown(
        holdings=holdings,
        analyses=analyses,
        hedge=hedge,
        candidates=candidates,
        errors=errors,
        skipped_valuation=skipped,
        warnings=warnings,
    )
    console.print(Markdown(md))
    write_failed = False
    if report is not None:
        try:
            report.write_text(md, encoding="utf-8")
            typer.echo(f"Report written to {report}", err=True)
        except OSError as exc:
            typer.echo(f"Error: could not write report to {report}: {exc}", err=True)
            write_failed = True
    if write_failed or not analyses:
        raise typer.Exit(1)
