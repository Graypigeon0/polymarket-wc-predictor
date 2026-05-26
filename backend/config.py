"""Centralised configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_service_key: str = Field(..., alias="SUPABASE_SERVICE_KEY")
    supabase_db_url: str | None = Field(None, alias="SUPABASE_DB_URL")

    # LLM (Mistral primary — EU-based, works in Cyprus and all EU countries)
    mistral_api_key: str = Field(..., alias="MISTRAL_API_KEY")
    mistral_model: str = Field("mistral-small-latest", alias="MISTRAL_MODEL")
    groq_api_key: str | None = Field(None, alias="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", alias="GROQ_MODEL")
    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", alias="GEMINI_MODEL")

    # Telegram
    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(..., alias="TELEGRAM_CHAT_ID")

    # Sports data
    football_data_api_key: str | None = Field(None, alias="FOOTBALL_DATA_API_KEY")

    # Polymarket
    polymarket_clob_url: str = Field(
        "https://clob.polymarket.com", alias="POLYMARKET_CLOB_URL"
    )

    # Runtime
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    environment: str = Field("development", alias="ENVIRONMENT")
    edge_alert_threshold: float = Field(0.0, alias="EDGE_ALERT_THRESHOLD")
    edge_alert_cooldown_minutes: int = Field(30, alias="EDGE_ALERT_COOLDOWN_MINUTES")

    # Modeling constants
    model_version: str = "0.1.0-dev"
    monte_carlo_runs: int = 10_000
    max_event_delta_pct: float = 0.05  # hard cap on per-event rating delta


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
