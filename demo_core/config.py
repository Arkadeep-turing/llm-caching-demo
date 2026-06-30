from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root without printing secrets.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class PricingConfig:
    input_price_per_mtok: float
    output_price_per_mtok: float
    cached_input_multiplier: float = 0.50  # OpenAI prompt caching discount (approx)
    embedding_price_per_mtok: float = 0.02


# Pricing map for common OpenAI models (USD per 1M tokens).
PRICING: dict[str, PricingConfig] = {
    "gpt-4o-mini": PricingConfig(
        input_price_per_mtok=0.15,
        output_price_per_mtok=0.60,
        cached_input_multiplier=0.50,
        embedding_price_per_mtok=0.02,
    ),
    "gpt-4o": PricingConfig(
        input_price_per_mtok=2.50,
        output_price_per_mtok=10.00,
        cached_input_multiplier=0.50,
        embedding_price_per_mtok=0.13,
    ),
    "gpt-4.1-mini": PricingConfig(
        input_price_per_mtok=0.40,
        output_price_per_mtok=1.60,
        cached_input_multiplier=0.50,
        embedding_price_per_mtok=0.02,
    ),
}


@dataclass
class DemoConfig:
    openai_api_key: str
    openai_model: str
    embedding_model: str
    redis_url: str
    temperature: float = 0.0
    max_output_tokens: int = 256
    semantic_threshold: float = 0.82
    exact_cache_ttl: int = 900
    context_cache_ttl: int = 600
    semantic_cache_ttl: int = 900
    prompt_cache_ttl: int = 600
    prompt_cache_min_tokens: int = 256  # Lower for live demo with smaller prompts

    @property
    def pricing(self) -> PricingConfig:
        return PRICING.get(self.openai_model, PRICING["gpt-4o-mini"])


def load_config() -> DemoConfig:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env in the project root."
        )
    return DemoConfig(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0").strip(),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
        max_output_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "256")),
    )


def check_redis_connection(redis_url: str) -> bool:
    import redis

    client = redis.from_url(redis_url, decode_responses=True)
    return bool(client.ping())
