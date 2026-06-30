from __future__ import annotations

import json
import time
from typing import Any, Optional

import numpy as np
import redis
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class RedisCacheBase:
    """Base Redis cache with namespaced keys and TTL."""

    def __init__(self, client: redis.Redis, namespace: str, ttl_seconds: int):
        self.client = client
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def clear_namespace(self) -> int:
        deleted = 0
        cursor = 0
        pattern = f"{self.namespace}:*"
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=200)
            if keys:
                deleted += self.client.delete(*keys)
            if cursor == 0:
                break
        return deleted


class ExactResponseCache(RedisCacheBase):
    """Exact-match response cache keyed by normalized prompt scope."""

    def get(self, key: str) -> Optional[dict[str, Any]]:
        raw = self.client.get(self._key(key))
        if not raw:
            return None
        return json.loads(raw)

    def set(self, key: str, answer: str, metadata: dict[str, Any]) -> None:
        payload = {"answer": answer, "metadata": metadata, "created_at": time.time()}
        self.client.setex(self._key(key), self.ttl_seconds, json.dumps(payload))


class ContextCache(RedisCacheBase):
    """Cache retrieved RAG context by intent and document version."""

    def get(self, key: str) -> Optional[dict[str, Any]]:
        raw = self.client.get(self._key(key))
        if not raw:
            return None
        return json.loads(raw)

    def set(self, key: str, context: str, metadata: dict[str, Any]) -> None:
        payload = {"context": context, "metadata": metadata, "created_at": time.time()}
        self.client.setex(self._key(key), self.ttl_seconds, json.dumps(payload))


class SemanticResponseCache(RedisCacheBase):
    """Embedding-based semantic response cache stored in Redis."""

    def __init__(
        self,
        client: redis.Redis,
        namespace: str,
        ttl_seconds: int,
        openai_client: OpenAI,
        embedding_model: str,
    ):
        super().__init__(client, namespace, ttl_seconds)
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self._index_key = f"{namespace}:index"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _embed(self, text: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def lookup(
        self, question: str, scope: str, threshold: float
    ) -> Optional[dict[str, Any]]:
        query_vec = self._embed(question)
        scope_key = f"{self._index_key}:{scope}"
        entry_ids = self.client.smembers(scope_key)
        best: Optional[dict[str, Any]] = None
        best_score = 0.0

        for entry_id in entry_ids:
            raw = self.client.get(self._key(entry_id))
            if not raw:
                self.client.srem(scope_key, entry_id)
                continue
            entry = json.loads(raw)
            score = self._cosine(query_vec, entry["embedding"])
            if score > best_score:
                best_score = score
                best = {**entry, "score": score}

        if best and best_score >= threshold:
            return best
        return None

    def add(
        self,
        question: str,
        scope: str,
        answer: str,
        metadata: dict[str, Any],
    ) -> None:
        entry_id = f"{scope}:{hash(question) & 0xFFFFFFFF:08x}:{int(time.time())}"
        embedding = self._embed(question)
        payload = {
            "question": question,
            "answer": answer,
            "embedding": embedding,
            "metadata": metadata,
            "scope": scope,
            "created_at": time.time(),
        }
        self.client.setex(self._key(entry_id), self.ttl_seconds, json.dumps(payload))
        self.client.sadd(f"{self._index_key}:{scope}", entry_id)
        self.client.expire(f"{self._index_key}:{scope}", self.ttl_seconds)


class PromptPrefixCache(RedisCacheBase):
    """
    Models provider-style prompt caching: tracks longest common prefix (by token count)
    across prior prompts in the same scope. Used to estimate cached billable tokens.
    """

    def __init__(
        self,
        client: redis.Redis,
        namespace: str,
        ttl_seconds: int,
        min_tokens: int = 256,
        increment: int = 64,
    ):
        super().__init__(client, namespace, ttl_seconds)
        self.min_tokens = min_tokens
        self.increment = increment

    def check_and_register(self, scope: str, prompt_tokens: list[str]) -> int:
        scope_key = self._key(f"scope:{scope}")
        raw = self.client.get(scope_key)
        now = time.time()
        best = 0
        sequences: list[list[str]] = []

        if raw:
            stored = json.loads(raw)
            for seq, expires_at in stored.get("sequences", []):
                if expires_at >= now:
                    sequences.append(seq)
                    lcp = 0
                    for a, b in zip(prompt_tokens, seq):
                        if a != b:
                            break
                        lcp += 1
                    best = max(best, lcp)

        if len(prompt_tokens) >= self.min_tokens:
            sequences.append(prompt_tokens)
        self.client.setex(
            scope_key,
            self.ttl_seconds,
            json.dumps({"sequences": [[s, now + self.ttl_seconds] for s in sequences[-20:]]}),
        )

        if best < self.min_tokens:
            return 0
        return (best // self.increment) * self.increment


class KVPrefixCache(RedisCacheBase):
    """
    Models inference-engine KV/prefix caching: saves prefill compute but not API billing.
    """

    def __init__(
        self,
        client: redis.Redis,
        namespace: str,
        ttl_seconds: int,
        block_size: int = 64,
    ):
        super().__init__(client, namespace, ttl_seconds)
        self.block_size = block_size

    def check_and_register(self, prompt_tokens: list[str]) -> int:
        key = self._key("global")
        raw = self.client.get(key)
        now = time.time()
        best = 0
        sequences: list[list[str]] = []

        if raw:
            stored = json.loads(raw)
            for seq, expires_at in stored.get("sequences", []):
                if expires_at >= now:
                    sequences.append(seq)
                    lcp = 0
                    for a, b in zip(prompt_tokens, seq):
                        if a != b:
                            break
                        lcp += 1
                    best = max(best, lcp)

        sequences.append(prompt_tokens)
        self.client.setex(
            key,
            self.ttl_seconds,
            json.dumps({"sequences": [[s, now + self.ttl_seconds] for s in sequences[-20:]]}),
        )
        return (best // self.block_size) * self.block_size


def create_redis_client(redis_url: str) -> redis.Redis:
    return redis.from_url(redis_url, decode_responses=True)


def clear_all_demo_caches(client: redis.Redis) -> None:
    """Clear all demo cache namespaces before a strategy run."""
    for ns in ("exact", "context", "semantic", "prompt_prefix", "kv_prefix"):
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor=cursor, match=f"{ns}:*", count=200)
            if keys:
                client.delete(*keys)
            if cursor == 0:
                break
