"""
Telegram alerter.

Pushes formatted edge alerts to the configured chat ID. One bot, one user.
"""

from __future__ import annotations

import structlog
from telegram import Bot
from telegram.constants import ParseMode

from backend.config import get_settings

log = structlog.get_logger()

_bot: Bot | None = None


def _get_bot() -> Bot:
    global _bot
    if _bot is None:
        _bot = Bot(token=get_settings().telegram_bot_token)
    return _bot


async def alert(
    market_label: str,
    market_url: str,
    model_prob: float,
    pm_prob: float,
    edge: float,
    driving_news: list[str] | None = None,
) -> bool:
    """Send a single edge alert. Returns True if sent successfully."""
    s = get_settings()
    driving = ""
    if driving_news:
        driving = "\n\n_Recent news driving model:_\n" + "\n".join(
            f"• {n}" for n in driving_news[:2]
        )

    msg = (
        f"🔥 *Edge detected*\n\n"
        f"*Market:* {market_label}\n"
        f"*Model:* {model_prob:.1%}\n"
        f"*Polymarket:* {pm_prob:.1%}\n"
        f"*Edge:* +{edge:.1%}\n\n"
        f"[Open on Polymarket]({market_url})"
        f"{driving}"
    )

    try:
        await _get_bot().send_message(
            chat_id=s.telegram_chat_id,
            text=msg,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
        log.info("telegram.alert.sent", market=market_label, edge=edge)
        return True
    except Exception as e:
        log.error("telegram.alert.failed", error=str(e))
        return False


async def heartbeat() -> None:
    """Send a daily heartbeat so you know the engine is alive."""
    await _get_bot().send_message(
        chat_id=get_settings().telegram_chat_id,
        text="✅ WC predictor engine alive",
    )
