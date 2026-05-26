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
    """Refresh squad data from Sofascore + FBref + football-data.org."""
    from backend.ingestion import fbref, football_data, sofascore

    log.info("refresh_squads.start")
    await sofascore.refresh_players()
    await fbref.refresh_underlying_stats()
    await football_data.refresh_fixtures()
    log.info("refresh_squads.done")


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
def backfill(competition: str = "EURO2024") -> None:
    """Backfill historical match data for a competition (for validation)."""
    typer.echo(f"Backfill for {competition} not yet implemented.")
    # TODO: pull historical results from football-data.org, populate matches table


if __name__ == "__main__":
    cli()
