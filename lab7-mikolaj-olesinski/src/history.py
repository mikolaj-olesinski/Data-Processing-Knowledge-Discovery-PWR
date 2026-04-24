import json
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from cache import get_redis
from pipeline import PipelineResult

HISTORY_KEY = "history:entries"
HISTORY_LIMIT = 500


def _serialize(result: PipelineResult) -> dict:
    entry = asdict(result)
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    entry["sources_count"] = len(result.sources)
    return entry


def save_query(result: PipelineResult) -> None:
    r = get_redis()
    entry = _serialize(result)
    r.rpush(HISTORY_KEY, json.dumps(entry, default=str))
    r.ltrim(HISTORY_KEY, -HISTORY_LIMIT, -1)


def load_history(limit: int = HISTORY_LIMIT) -> list[dict]:
    r = get_redis()
    raw = r.lrange(HISTORY_KEY, -limit, -1)
    return [json.loads(x) for x in raw]


def get_recent(limit: int = 10) -> list[dict]:
    return list(reversed(load_history(limit)))


def load_history_df() -> pd.DataFrame:
    entries = load_history()
    if not entries:
        return pd.DataFrame(columns=[
            "timestamp", "topic", "original_chars", "summary_chars",
            "llm_seconds", "total_seconds", "sources_count", "errors_count", "cache_hit",
        ])
    df = pd.DataFrame(entries)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clear_history() -> int:
    r = get_redis()
    return int(r.delete(HISTORY_KEY))
