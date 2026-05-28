"""
Main entry point for the prediction engine.

Runs the scheduler that orchestrates all ingestion, modeling, and alerting jobs.
Also exposes a Typer CLI for one-shot operations (backfill, fit model, simulate).
"""

from __future__ import annotations

import asyncio
import signal
import sys

import structlog
import typer
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.config import get_settings

cli = typer.Typer(help="World Cup 2026 Polymarket prediction engine")
log = structlog.get_logger()


# ---------------------------------------------------------------------
# Scheduled job stubs - each is filled in by the matching module
# ---------------------------------------------------------------------

async def poll_news() -> None:
    """Pull latest RSS + Reddit, classify with Gemini, apply rating deltas."""
    from backend.ingestion.news import rss, reddit
    from backend.models import llm_context

    log.info("poll_news.start")
    items = await rss.fetch_all()
    items += await reddit.fetch_all()
    await llm_context.classify_and_adjust(items)
    log.info("poll_news.done", count=len(items))


async def poll_polymarket() -> None:
    """Refresh Polymarket prices for tracked WC markets and recompute edges."""
    from backend.edges import calculator
    from backend.ingestion import polymarket

    log.info("poll_polymarket.start")
    await polymarket.refresh_prices()
    await calculator.recompute_all()
    log.info("poll_polymarket.done")


async def refresh_squads() -> None:
    """Refresh fixtures, then squad/player data from Sofascore + FBref."""
    from backend.ingestion import fbref, football_data, sofascore

    log.info("refresh_squads.start")
    fixtures_written = await football_data.refresh_fixtures()
    await sofascore.refresh_players()
    await fbref.refresh_underlying_stats()
    log.info("refresh_squads.done", fixtures=fixtures_written)


async def run_match_models() -> None:
    """Recompute match-level predictions for upcoming WC fixtures."""
    from backend.models import dixon_coles, squad_strength

    log.info("run_match_models.start")
    await squad_strength.recompute_team_ratings()
    await dixon_coles.predict_upcoming()
    log.info("run_match_models.done")


async def run_tournament_sim() -> None:
    """Run Monte Carlo tournament sim and top-scorer model."""
    from backend.models import top_scorer, tournament_sim

    log.info("run_tournament_sim.start")
    await tournament_sim.simulate()
    await top_scorer.predict()
    log.info("run_tournament_sim.done")


# ---------------------------------------------------------------------
# Scheduler wiring
# ---------------------------------------------------------------------

def build_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()

    # Fast loops
    scheduler.add_job(poll_polymarket, IntervalTrigger(seconds=60),
                      id="poll_polymarket", max_instances=1)
    scheduler.add_job(poll_news, IntervalTrigger(minutes=5),
                      id="poll_news", max_instances=1)

    # Medium
    scheduler.add_job(refresh_squads, IntervalTrigger(hours=1),
                      id="refresh_squads", max_instances=1)
    scheduler.add_job(run_match_models, IntervalTrigger(minutes=15),
                      id="run_match_models", max_instances=1)

    # Slow / daily
    scheduler.add_job(run_tournament_sim, CronTrigger(hour=4, minute=0),
                      id="run_tournament_sim", max_instances=1)

    return scheduler


async def serve() -> None:
    settings = get_settings()
    log.info("engine.starting", environment=settings.environment,
             model_version=settings.model_version)

    scheduler = build_scheduler()
    scheduler.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    scheduler.shutdown(wait=True)
    log.info("engine.stopped")


# ---------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------

@cli.command()
def run() -> None:
    """Run the scheduler (long-lived)."""
    asyncio.run(serve())


@cli.command()
def once(job: str) -> None:
    """Run a single job once and exit. Choices: news, polymarket, squads, match, sim."""
    jobs = {
        "news": poll_news,
        "polymarket": poll_polymarket,
        "squads": refresh_squads,
        "match": run_match_models,
        "sim": run_tournament_sim,
    }
    fn = jobs.get(job)
    if fn is None:
        typer.echo(f"Unknown job: {job}. Choose from {list(jobs)}.", err=True)
        sys.exit(1)
    asyncio.run(fn())


@cli.command()
def backfill(competition: str = "WC2022") -> None:
    """Backfill one competition. WC2026 hits football-data.org (live data),
    everything else uses openfootball (historical)."""
    if competition == "WC2026":
        from backend.ingestion import football_data
        written = asyncio.run(football_data.refresh_competition(competition))
    else:
        from backend.ingestion import openfootball
        written = asyncio.run(openfootball.refresh_competition(competition))
    typer.echo(f"Backfill complete for {competition}: {written} matches written.")


@cli.command(name="backfill-all")
def backfill_all() -> None:
    """Pull WC2026 fixtures + all historical competitions in one go."""
    from backend.ingestion import football_data, openfootball
    live = asyncio.run(football_data.refresh_fixtures())
    historical = asyncio.run(openfootball.refresh_all())
    typer.echo(f"Backfill complete: {live} live + {historical} historical = {live + historical} matches.")


@cli.command(name="db-test")
def db_test() -> None:
    """Diagnostic: insert a test team into Supabase and read it back."""
    import uuid as _uuid
    from backend.db.client import get_client

    db = get_client()
    test_code = "ZZT"  # unlikely to collide

    # Clean up any prior test row
    try:
        db.table("teams").delete().eq("fifa_code", test_code).execute()
    except Exception as e:
        typer.echo(f"Cleanup warning: {e}")

    # Insert
    team_id = str(_uuid.uuid4())
    try:
        r = db.table("teams").insert({
            "id": team_id,
            "fifa_code": test_code,
            "name": "DB Test Team",
            "confederation": "TEST",
        }).execute()
        typer.echo(f"INSERT OK -> {len(r.data)} row(s) returned")
    except Exception as e:
        typer.echo(f"INSERT FAILED: {type(e).__name__}: {e}", err=True)
        return

    # Read back
    try:
        r = db.table("teams").select("*").eq("fifa_code", test_code).execute()
        typer.echo(f"SELECT OK -> {r.data}")
    except Exception as e:
        typer.echo(f"SELECT FAILED: {type(e).__name__}: {e}", err=True)

    # Clean up
    db.table("teams").delete().eq("fifa_code", test_code).execute()
    typer.echo("Cleanup complete.")



@cli.command()
def fit() -> None:
    """Fit Dixon-Coles base ratings on historical matches; persist to DB."""
    from backend.models import dixon_coles
    result = asyncio.run(dixon_coles.fit())
    typer.echo(f"Fit result: {result}")


@cli.command()
def predict() -> None:
    """Generate match_predictions rows for all upcoming WC fixtures."""
    from backend.models import dixon_coles
    n = asyncio.run(dixon_coles.predict_upcoming())
    typer.echo(f"Wrote {n} predictions.")



@cli.command(name="backfill-martj42")
def backfill_martj42(years: int = 5) -> None:
    """Pull last N years of international results from martj42 dataset."""
    from backend.ingestion import martj42
    n = asyncio.run(martj42.refresh(lookback_years=years))
    typer.echo(f"martj42 backfill complete: {n} matches written (last {years} years).")



@cli.command(name="pm-discover")
def pm_discover() -> None:
    """Scan Polymarket and register all active WC 2026 markets in our DB."""
    from backend.ingestion import polymarket
    result = asyncio.run(polymarket.discover_wc_markets())
    typer.echo(f"Discovery result: {result}")


@cli.command(name="pm-refresh")
def pm_refresh() -> None:
    """Pull latest prices for all tracked Polymarket markets."""
    from backend.ingestion import polymarket
    n = asyncio.run(polymarket.refresh_prices())
    typer.echo(f"Refreshed {n} market prices.")


@cli.command(name="edges")
def edges_cmd() -> None:
    """Compute edges and fire Telegram alerts where positive."""
    from backend.edges import calculator
    stats = asyncio.run(calculator.recompute_all())
    typer.echo(f"Edges: {stats}")



@cli.command(name="sim")
def sim() -> None:
    """Run the Monte Carlo tournament simulation; persist outright/group probs."""
    from backend.models import tournament_sim
    result = asyncio.run(tournament_sim.simulate())
    typer.echo(f"Tournament sim: {result}")



@cli.command()
def backtest(target: str = "EURO2024") -> None:
    """Backtest model calibration on a past tournament (EURO2024, WC2022, COPA2024)."""
    from backend.models import backtest as bt
    result = asyncio.run(bt.run_backtest(target))
    import json
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
