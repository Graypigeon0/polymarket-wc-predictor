"""
Edge calculator.

For each tracked Polymarket market, compare our model probability to the
current Polymarket implied probability and flag any positive-edge opportunities.

edge = model_prob - pm_prob

Alerts fire when edge > settings.edge_alert_threshold AND lower CI bound is also positive.
"""

from __future__ import annotations

import structlog

from backend.alerts import telegram
from backend.config import get_settings

log = structlog.get_logger()


async def recompute_all() -> None:
    """
    Pull every active Polymarket market, look up matching model prediction,
    compute edge + confidence interval, write to `edges` table, fire alerts.
    """
    s = get_settings()
    log.info("edges.recompute.todo", threshold=s.edge_alert_threshold)
    # TODO:
    #   1. SELECT * FROM polymarket_markets WHERE active = true
    #   2. For each market:
    #      - look up model_prob from match_predictions / tournament_predictions /
    #        top_scorer_predictions depending on market_type
    #      - look up latest pm_price for token_id -> pm_prob = price
    #      - edge = model_prob - pm_prob
    #      - lower_ci = edge - 1.96 * model_std (rough confidence band)
    #   3. INSERT edges row
    #   4. If edge > threshold AND lower_ci > 0 AND not within cooldown of last alert:
    #        await telegram.alert(market, model_prob, pm_prob, edge)
    #        mark alerted=true


def implied_prob_from_price(price: float) -> float:
    """Polymarket prices are already 0..1 probabilities (USDC of payout per $1 stake)."""
    return max(0.0, min(1.0, price))
