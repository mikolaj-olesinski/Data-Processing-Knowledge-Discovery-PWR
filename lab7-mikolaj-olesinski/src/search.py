import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException

from cache import redis_cache
from metrics import ERRORS

SEARCH_TTL = 24 * 3600
BACKENDS = ("duckduckgo", "google", "bing")


class SearchError(Exception):
    pass


def _try_search(topic: str, n: int, region: str) -> list[dict]:
    last_err: Exception | None = None
    for backend in BACKENDS:
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    return list(ddgs.text(
                        topic,
                        region=region,
                        safesearch="moderate",
                        max_results=n * 2,
                        backend=backend,
                    ))
            except RatelimitException as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
            except DDGSException as e:
                last_err = e
                break
    ERRORS.labels(kind="search").inc()
    raise SearchError(f"DDGS rate-limit/blad na wszystkich backendach: {last_err}")


@redis_cache(ttl=SEARCH_TTL, key_prefix="search")
def search_top(topic: str, n: int = 3, region: str = "pl-pl") -> list[dict]:
    topic = topic.strip()
    if not topic:
        raise SearchError("Pusty temat wyszukiwania.")

    results = _try_search(topic, n, region)

    cleaned: list[dict] = []
    seen: set[str] = set()
    for r in results:
        url = r.get("href") or r.get("url") or ""
        title = r.get("title") or ""
        body = r.get("body") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        cleaned.append({"url": url, "title": title, "snippet": body})
        if len(cleaned) >= n:
            break

    if not cleaned:
        raise SearchError(f"Brak wynikow wyszukiwania dla: {topic}")
    return cleaned
