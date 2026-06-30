# Live Context Caching & AI Cost Optimization Demo

This project runs **real OpenAI API calls** with **Redis-backed caches** and **LangChain orchestration** to demonstrate how caching and token optimization reduce latency and cost.

The original mock notebook (`context_caching_ai_cost_optimization_demo.ipynb`) is unchanged. The live version is in `live_context_caching_ai_cost_optimization_demo.ipynb`.

## What this demo uses (real components)

| Layer | Technology |
|-------|------------|
| LLM | OpenAI via LangChain `ChatOpenAI` |
| Exact / context cache | Redis with TTL |
| Semantic cache | OpenAI embeddings + Redis |
| Token counting | `tiktoken` |
| Prompt/KV prefix cache | Redis-tracked prefix modeling |

## Quick start

```bash
# 1. Start Redis
docker compose up -d

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 4. Smoke test (4 requests, ~$0.01 on gpt-4o-mini)
DEMO_ROUNDS=1 python smoke_test.py

# 5. Open the live notebook
jupyter notebook live_context_caching_ai_cost_optimization_demo.ipynb
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for completions |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model for semantic cache |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `DEMO_ROUNDS` | `3` | Workload rounds (18 questions each) |

## Project structure

```
Caching demo/
├── demo_core/           # Reusable Python modules
│   ├── config.py        # Env loading, pricing
│   ├── workload.py      # Request workload + prompts
│   ├── redis_cache.py   # Redis cache implementations
│   ├── llm_runner.py    # LangChain + OpenAI pipeline
│   └── metrics.py       # Token counting + cost aggregation
├── docker-compose.yml   # Local Redis
├── smoke_test.py        # Minimal end-to-end test
└── live_context_caching_ai_cost_optimization_demo.ipynb
```

## Cost note

Default workload: **3 rounds × 18 questions = 54 requests**. At ~800 input + 150 output tokens each on `gpt-4o-mini`, expect roughly **$0.01–0.05** depending on cache hit rates.

For a quick demo, set `DEMO_ROUNDS=1` (~18 requests).

## Notes

- **tiktoken** downloads encoding files on first use (needs internet once).
- Provider prompt caching and KV prefix caching are **modeled** via Redis prefix tracking; OpenAI does not expose these as direct API toggles.
- Semantic cache uses real OpenAI embeddings — each lookup costs a small embedding API call on cache miss.
