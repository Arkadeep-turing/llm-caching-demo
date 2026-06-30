from __future__ import annotations

from typing import Optional

import pandas as pd
import tiktoken

from demo_core.config import PricingConfig


def get_tokenizer(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> int:
    enc = get_tokenizer(model)
    return len(enc.encode(text))


def tokenize(text: str, model: str) -> list[str]:
    enc = get_tokenizer(model)
    return [str(t) for t in enc.encode(text)]


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: PricingConfig,
    cached_billable_tokens: int = 0,
    embedding_tokens: int = 0,
) -> float:
    bill_cached = min(cached_billable_tokens, input_tokens)
    uncached = input_tokens - bill_cached
    llm_cost = (
        uncached * pricing.input_price_per_mtok
        + bill_cached * pricing.input_price_per_mtok * pricing.cached_input_multiplier
        + output_tokens * pricing.output_price_per_mtok
    ) / 1_000_000
    embed_cost = (embedding_tokens * pricing.embedding_price_per_mtok) / 1_000_000
    return llm_cost + embed_cost


def summarize_results(df: pd.DataFrame, baseline_name: str = "Baseline") -> pd.DataFrame:
    def p95(x: pd.Series) -> float:
        return float(x.quantile(0.95))

    out = df.groupby("strategy").agg(
        requests=("req_id", "count"),
        llm_calls=("source", lambda s: int((s == "llm").sum())),
        total_cost_usd=("cost_usd", "sum"),
        avg_latency_ms=("latency_ms", "mean"),
        p95_latency_ms=("latency_ms", p95),
        total_input_tokens=("input_tokens", "sum"),
        total_output_tokens=("output_tokens", "sum"),
        cached_billable_tokens=("cached_billable_tokens", "sum"),
        cached_compute_tokens=("cached_compute_tokens", "sum"),
        cache_hits=("cache_event", lambda s: int((s != "miss").sum())),
        stale_hits=("stale_hit", "sum"),
        semantic_false_positive=("semantic_false_positive", "sum"),
    ).reset_index()

    base = out.loc[out["strategy"] == baseline_name, ["total_cost_usd", "avg_latency_ms"]]
    if not base.empty:
        base_cost = float(base["total_cost_usd"].iloc[0])
        base_lat = float(base["avg_latency_ms"].iloc[0])
        out["cost_reduction_%"] = (1 - out["total_cost_usd"] / base_cost) * 100
        out["latency_reduction_%"] = (1 - out["avg_latency_ms"] / base_lat) * 100
    return out.sort_values("total_cost_usd").reset_index(drop=True)


def format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    display_cols = [
        "strategy",
        "requests",
        "llm_calls",
        "total_cost_usd",
        "avg_latency_ms",
        "p95_latency_ms",
        "cached_billable_tokens",
        "cached_compute_tokens",
        "cache_hits",
        "cost_reduction_%",
        "latency_reduction_%",
        "stale_hits",
    ]
    cols = [c for c in display_cols if c in summary.columns]
    return summary[cols].copy()


def format_summary_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with human-readable formatting (no jinja2 / Styler required)."""
    out = format_summary_table(summary).copy()
    if "total_cost_usd" in out.columns:
        out["total_cost_usd"] = out["total_cost_usd"].map(lambda x: f"${x:.4f}")
    if "avg_latency_ms" in out.columns:
        out["avg_latency_ms"] = out["avg_latency_ms"].map(lambda x: f"{x:.0f}")
    if "p95_latency_ms" in out.columns:
        out["p95_latency_ms"] = out["p95_latency_ms"].map(lambda x: f"{x:.0f}")
    if "cost_reduction_%" in out.columns:
        out["cost_reduction_%"] = out["cost_reduction_%"].map(lambda x: f"{x:.1f}")
    if "latency_reduction_%" in out.columns:
        out["latency_reduction_%"] = out["latency_reduction_%"].map(lambda x: f"{x:.1f}")
    return out
