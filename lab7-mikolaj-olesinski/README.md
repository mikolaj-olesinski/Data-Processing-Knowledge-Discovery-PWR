# Web Summarizer — LLM-Powered Topic Summarization

A web application that takes any topic, automatically searches the web for the best
articles, scrapes them, and generates a coherent summary using an LLM.
A production-ready setup with containerization, multi-layer caching, and full monitoring.

Built for the PWDIOW course (assignment 7).

## Stack

- **Frontend:** Streamlit (chat UI, tabs, progress bar)
- **LLM:** Google Gemini (`gemini-2.5-flash-lite`) via LangChain
- **Search & scraping:** DuckDuckGo (`ddgs`) with retry/backoff, trafilatura
- **Caching & state:** Redis (scrape, search, LLM cache + query history)
- **Monitoring:** Prometheus + Grafana (dashboards & alerting)
- **Charts:** Plotly
- **Infra:** Docker + Docker Compose

## How it works

```
user → Streamlit → search (DuckDuckGo) → scrape (trafilatura) → LLM summary (map-reduce)
                        ↓
                     Redis (cache + history)
                        ↓
                  Prometheus (/metrics) → Grafana (dashboards + alerts)
```

The summarization uses a **map-reduce** strategy: combined article text can reach
100–200k characters, so the chain first extracts facts from each chunk, then merges
everything into one coherent summary.

## Features

- **Topic input** with a slider for the number of sources (2–5)
- **Web search** via DuckDuckGo with retry + backoff
- **Article scraping** with trafilatura
- **Chat with follow-ups** — LLM keeps full dialogue context (LangChain message history)
- **Multi-layer Redis cache** — scrape (7 days), search (1 day), LLM responses
  (LangChain `RedisCache`); the same topic returns in <0.3 s with no API/token cost
- **Query history** in the sidebar (last queries persisted in Redis, restorable without re-querying)
- **Progress bar** — Search → Scrape → AI generation → Done
- **5 Plotly charts** — original vs summary length, LLM response times, OK/failed sources,
  cache hit rate, per-session LLM times

## Production Docker image

- Minimal (python-slim), complete (no external mounts required)
- No credentials baked into the image — `GOOGLE_API_KEY` passed via environment variable
- `.env.example` provided as a template (no secrets)

## Monitoring (Prometheus + Grafana)

Three metrics exported from the app:

1. `lab7_llm_seconds` — histogram of Gemini response time
2. `lab7_cache_events_total` — cache hits/misses (labels: `layer`, `result`)
3. `lab7_errors_total` — pipeline errors (label: `kind`)

Grafana dashboard (auto-provisioned) shows LLM p50/p95 latency, cache rate/min, and
error rate/min. An **alert** fires when `rate(lab7_errors_total[1m]) > 0.05` for 2 minutes.

## Running

```bash
cp .env.example .env
# add your GOOGLE_API_KEY to .env
docker compose up -d --build
```

- App: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (anonymous viewer access)
