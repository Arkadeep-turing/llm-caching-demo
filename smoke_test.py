"""Smoke test for live caching demo — run with DEMO_ROUNDS=1 for minimal API usage."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo_core.config import check_redis_connection, load_config
from demo_core.llm_runner import LiveLLMRunner
from demo_core.metrics import format_summary_table, summarize_results
from demo_core.redis_cache import create_redis_client
from demo_core.workload import Request, StrategyConfig, make_workload


def make_smoke_requests() -> list[Request]:
    """Minimal workload with deliberate duplicates so exact cache can be verified."""
    duplicate = "Can I get a refund if my shipment arrived late?"
    return [
        Request(0, duplicate, "refund", "user-0", "acme", 0),
        Request(1, "How do returns work when delivery was delayed?", "refund", "user-1", "acme", 18),
        Request(2, duplicate, "refund", "user-2", "acme", 36),
        Request(3, duplicate, "refund", "user-3", "acme", 54),
    ]


def main() -> None:
    config = load_config()
    assert check_redis_connection(config.redis_url), "Redis not reachable"
    print("✓ Redis connected")

    runner = LiveLLMRunner(config, create_redis_client(config.redis_url))
    requests = make_smoke_requests()
    print(f"✓ Workload: {len(requests)} requests (2 unique, 2 exact duplicates)")

    baseline = StrategyConfig("Baseline")
    exact = StrategyConfig("Exact response reuse", exact_response_cache=True)

    baseline_df = runner.run_strategy(baseline, requests, clear_caches=True)
    exact_df = runner.run_strategy(exact, requests, clear_caches=True)

    assert (baseline_df["source"] == "llm").all(), "Baseline should call LLM for all requests"
    assert (exact_df["source"] == "llm").sum() < len(exact_df), "Exact cache should hit on repeats"

    combined = summarize_results(
        __import__("pandas").concat([baseline_df, exact_df], ignore_index=True),
        baseline_name="Baseline",
    )
    print("\nSummary:")
    print(format_summary_table(combined).to_string(index=False))
    print("\n✓ Smoke test passed")


if __name__ == "__main__":
    main()
