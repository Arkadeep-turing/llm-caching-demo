from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd
import redis
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from demo_core.config import DemoConfig
from demo_core.metrics import count_tokens, estimate_cost, tokenize
from demo_core.redis_cache import (
    ContextCache,
    ExactResponseCache,
    KVPrefixCache,
    PromptPrefixCache,
    SemanticResponseCache,
    clear_all_demo_caches,
    create_redis_client,
)
from demo_core.workload import (
    Request,
    StrategyConfig,
    build_prompt,
    normalize_text,
    retrieve_context,
    stable_hash,
)


class LiveLLMRunner:
    """LangChain-based LLM runner with Redis-backed caching layers."""

    def __init__(self, config: DemoConfig, redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.redis = redis_client or create_redis_client(config.redis_url)
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.llm = ChatOpenAI(
            model=config.openai_model,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            api_key=config.openai_api_key,
        )
        self.exact_cache = ExactResponseCache(self.redis, "exact", config.exact_cache_ttl)
        self.context_cache = ContextCache(self.redis, "context", config.context_cache_ttl)
        self.semantic_cache = SemanticResponseCache(
            self.redis,
            "semantic",
            config.semantic_cache_ttl,
            self.openai_client,
            config.embedding_model,
        )
        self.prompt_cache = PromptPrefixCache(
            self.redis,
            "prompt_prefix",
            config.prompt_cache_ttl,
            min_tokens=config.prompt_cache_min_tokens,
        )
        self.kv_cache = KVPrefixCache(self.redis, "kv_prefix", config.prompt_cache_ttl)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _invoke_llm(self, prompt: str) -> dict[str, Any]:
        start = time.perf_counter()
        messages = [
            SystemMessage(content="You are Acme Support Assistant. Answer concisely using the provided context."),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        elapsed_ms = (time.perf_counter() - start) * 1000
        answer = response.content if isinstance(response.content, str) else str(response.content)

        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens") or count_tokens(prompt, self.config.openai_model)
        output_tokens = usage.get("output_tokens") or count_tokens(answer, self.config.openai_model)

        return {
            "answer": answer,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "latency_ms": elapsed_ms,
        }

    def run_strategy(
        self,
        strategy: StrategyConfig,
        requests: list[Request],
        clear_caches: bool = True,
    ) -> pd.DataFrame:
        if clear_caches:
            clear_all_demo_caches(self.redis)

        rows: list[dict[str, Any]] = []
        pricing = self.config.pricing

        for i, req in enumerate(requests):
            docs_version = 2 if strategy.update_docs_at is not None and i >= strategy.update_docs_at else 1
            prompt_version = "v2-optimized" if strategy.token_optimization else "v1-full"
            version_scope = (
                f"docs:{docs_version}|prompt:{prompt_version}"
                if strategy.include_versions_in_keys
                else "docs:any|prompt:any"
            )
            base_scope = f"tenant:{req.tenant_id}|model:{self.config.openai_model}|{version_scope}"
            cache_event = "miss"
            cached_billable_tokens = 0
            cached_compute_tokens = 0
            stale_hit = False
            semantic_false_positive = False
            embedding_tokens = 0
            retrieval_latency_ms = 95.0

            # 1) Exact response cache
            exact_key = stable_hash(base_scope, normalize_text(req.question), "temperature:0")
            if strategy.exact_response_cache:
                hit = self.exact_cache.get(exact_key)
                if hit:
                    metadata = hit.get("metadata", {})
                    stale_hit = metadata.get("docs_version") != docs_version
                    rows.append(
                        _result_row(
                            strategy.name,
                            req,
                            source="exact_response_cache",
                            cache_event="exact_hit",
                            latency_ms=12.0,
                            cost_usd=0.0,
                            docs_version=docs_version,
                            answer_preview=hit["answer"][:80],
                            stale_hit=stale_hit,
                        )
                    )
                    continue

            # 2) Semantic response cache
            if strategy.semantic_response_cache:
                threshold = strategy.semantic_threshold or self.config.semantic_threshold
                semantic_hit = self.semantic_cache.lookup(req.question, base_scope, threshold)
                if semantic_hit:
                    metadata = semantic_hit.get("metadata", {})
                    semantic_false_positive = metadata.get("intent") != req.intent
                    stale_hit = metadata.get("docs_version") != docs_version
                    embedding_tokens = count_tokens(req.question, self.config.embedding_model)
                    rows.append(
                        _result_row(
                            strategy.name,
                            req,
                            source="semantic_response_cache",
                            cache_event=f"semantic_hit@{semantic_hit['score']:.2f}",
                            latency_ms=16.0,
                            cost_usd=estimate_cost(0, 0, pricing, embedding_tokens=embedding_tokens),
                            docs_version=docs_version,
                            answer_preview=semantic_hit["answer"][:80],
                            stale_hit=stale_hit,
                            semantic_false_positive=semantic_false_positive,
                        )
                    )
                    continue

            # 3) Context cache (RAG retrieval)
            ctx_key = f"tenant:{req.tenant_id}|intent:{req.intent}|{version_scope}"
            context_hit = False
            if strategy.context_cache:
                ctx_entry = self.context_cache.get(ctx_key)
                if ctx_entry:
                    context = ctx_entry["context"]
                    context_hit = True
                    retrieval_latency_ms = 7.0
                else:
                    context = retrieve_context(req.intent, docs_version, strategy.token_optimization)
                    self.context_cache.set(
                        ctx_key,
                        context,
                        {"docs_version": docs_version, "intent": req.intent},
                    )
            else:
                context = retrieve_context(req.intent, docs_version, strategy.token_optimization)

            prompt = build_prompt(
                req,
                context,
                optimized=strategy.token_optimization,
                cache_friendly_order=strategy.cache_friendly_order,
            )
            prompt_token_list = tokenize(prompt, self.config.openai_model)

            # 4) Provider-style prompt prefix cache (modeled)
            prompt_scope = f"tenant:{req.tenant_id}|model:{self.config.openai_model}|prompt:{prompt_version}"
            if strategy.include_versions_in_keys:
                prompt_scope += f"|docs:{docs_version}"
            if strategy.prompt_cache:
                cached_billable_tokens = self.prompt_cache.check_and_register(
                    prompt_scope, prompt_token_list
                )

            # 5) KV prefix cache (modeled compute savings)
            if strategy.kv_prefix_cache:
                cached_compute_tokens = self.kv_cache.check_and_register(prompt_token_list)

            # 6) Real LLM call via LangChain
            llm_result = self._invoke_llm(prompt)
            total_latency = llm_result["latency_ms"] + retrieval_latency_ms
            answer = llm_result["answer"]

            cost = estimate_cost(
                llm_result["input_tokens"],
                llm_result["output_tokens"],
                pricing,
                cached_billable_tokens=cached_billable_tokens,
            )

            # Adjust cost if prompt cache discount applies (modeled layer)
            if cached_billable_tokens > 0:
                cache_event = "prompt_cache_hit"
            elif cached_compute_tokens > 0:
                cache_event = "kv_prefix_hit"
            elif context_hit:
                cache_event = "context_hit"

            # Populate caches after LLM call
            if strategy.exact_response_cache:
                self.exact_cache.set(
                    exact_key,
                    answer,
                    {"docs_version": docs_version, "intent": req.intent, "prompt_version": prompt_version},
                )
            if strategy.semantic_response_cache:
                self.semantic_cache.add(
                    req.question,
                    base_scope,
                    answer,
                    {"docs_version": docs_version, "intent": req.intent, "prompt_version": prompt_version},
                )

            rows.append(
                _result_row(
                    strategy.name,
                    req,
                    source="llm",
                    cache_event=cache_event,
                    latency_ms=total_latency,
                    cost_usd=cost,
                    input_tokens=llm_result["input_tokens"],
                    output_tokens=llm_result["output_tokens"],
                    cached_billable_tokens=cached_billable_tokens,
                    cached_compute_tokens=max(cached_compute_tokens, cached_billable_tokens),
                    context_cache_hit=context_hit,
                    docs_version=docs_version,
                    answer_preview=answer[:80],
                    stale_hit=stale_hit,
                    semantic_false_positive=semantic_false_positive,
                )
            )

        return pd.DataFrame(rows)


def _result_row(
    strategy: str,
    req: Request,
    source: str,
    cache_event: str,
    latency_ms: float,
    cost_usd: float,
    docs_version: int,
    answer_preview: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_billable_tokens: int = 0,
    cached_compute_tokens: int = 0,
    context_cache_hit: bool = False,
    stale_hit: bool = False,
    semantic_false_positive: bool = False,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "req_id": req.req_id,
        "intent": req.intent,
        "source": source,
        "cache_event": cache_event,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_billable_tokens": cached_billable_tokens,
        "cached_compute_tokens": cached_compute_tokens,
        "context_cache_hit": context_cache_hit,
        "stale_hit": stale_hit,
        "semantic_false_positive": semantic_false_positive,
        "docs_version": docs_version,
        "answer_preview": answer_preview,
    }
