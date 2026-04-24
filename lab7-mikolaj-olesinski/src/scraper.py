import trafilatura

from cache import redis_cache

SCRAPE_TTL = 7 * 24 * 3600


class ScrapeError(Exception):
    pass


@redis_cache(ttl=SCRAPE_TTL, key_prefix="scrape")
def fetch_text(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ScrapeError(f"Nie udalo sie pobrac strony: {url}")
    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not extracted:
        raise ScrapeError(f"Nie udalo sie wyodrebnic tekstu z: {url}")
    return extracted
