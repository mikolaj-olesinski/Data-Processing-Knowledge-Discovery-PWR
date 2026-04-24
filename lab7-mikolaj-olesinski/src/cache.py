import functools
import hashlib
import json
from typing import Callable

import redis

from config import get_redis_url
from metrics import CACHE_EVENTS

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(get_redis_url(), decode_responses=True)
    return _client


def ping() -> bool:
    try:
        return bool(get_redis().ping())
    except redis.exceptions.RedisError:
        return False


def _make_key(prefix: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps([args, kwargs], sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:32]
    return f"{prefix}:{digest}"


def redis_cache(ttl: int, key_prefix: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            r = get_redis()
            key = _make_key(key_prefix, args, kwargs)
            cached = r.get(key)
            if cached is not None:
                CACHE_EVENTS.labels(layer=key_prefix, result="hit").inc()
                return json.loads(cached)
            CACHE_EVENTS.labels(layer=key_prefix, result="miss").inc()
            result = fn(*args, **kwargs)
            r.setex(key, ttl, json.dumps(result, default=str))
            return result
        return wrapper
    return decorator
