from prometheus_client import Counter, Histogram, start_http_server

LLM_SECONDS = Histogram(
    "lab7_llm_seconds",
    "Czas odpowiedzi LLM (Gemini) w sekundach",
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

CACHE_EVENTS = Counter(
    "lab7_cache_events_total",
    "Trafienia i pudla cache wg warstwy",
    ["layer", "result"],
)

ERRORS = Counter(
    "lab7_errors_total",
    "Bledy pipeline'u wg typu",
    ["kind"],
)

_started = False


def start_metrics_server(port: int = 9100) -> None:
    global _started
    if not _started:
        start_http_server(port)
        _started = True
