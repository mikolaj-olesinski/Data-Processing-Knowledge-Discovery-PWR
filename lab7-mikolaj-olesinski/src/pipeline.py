import time
from dataclasses import dataclass, field
from typing import Callable

from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from llm import get_llm
from metrics import CACHE_EVENTS, ERRORS, LLM_SECONDS
from scraper import ScrapeError, fetch_text
from search import search_top

ProgressCb = Callable[[str, float], None]
CACHE_HIT_THRESHOLD_S = 0.3
MAX_CHARS_PER_DOC = 60_000


@dataclass
class SourceResult:
    url: str
    title: str
    chars: int = 0
    error: str | None = None


@dataclass
class PipelineResult:
    topic: str
    summary: str = ""
    sources: list[SourceResult] = field(default_factory=list)
    original_chars: int = 0
    summary_chars: int = 0
    llm_seconds: float = 0.0
    total_seconds: float = 0.0
    tokens: dict = field(default_factory=dict)
    cache_hit: bool = False
    errors_count: int = 0
    context_docs: list[dict] = field(default_factory=list)

    @classmethod
    def from_entry(cls, entry: dict) -> "PipelineResult":
        sources = [SourceResult(**s) for s in entry.get("sources", [])]
        return cls(
            topic=entry.get("topic", ""),
            summary=entry.get("summary", ""),
            sources=sources,
            original_chars=entry.get("original_chars", 0),
            summary_chars=entry.get("summary_chars", 0),
            llm_seconds=entry.get("llm_seconds", 0.0),
            total_seconds=entry.get("total_seconds", 0.0),
            tokens=entry.get("tokens", {}) or {},
            cache_hit=entry.get("cache_hit", False),
            errors_count=entry.get("errors_count", 0),
            context_docs=entry.get("context_docs", []) or [],
        )


MAP_PROMPT = PromptTemplate.from_template(
    "Wypisz kluczowe fakty z tego fragmentu w 3-5 punktach, po polsku:\n\n{text}"
)
COMBINE_PROMPT = PromptTemplate.from_template(
    "Na podstawie ponizszych notatek z roznych zrodel stworz spojne, zwiezle podsumowanie w 6-10 zdaniach po polsku. "
    "Unikaj powtorzen, zacznij od najwazniejszych fakow:\n\n{text}"
)


def _noop(_stage: str, _pct: float) -> None:
    pass


def summarize_topic(topic: str, n_sources: int = 3, progress_cb: ProgressCb | None = None) -> PipelineResult:
    cb = progress_cb or _noop
    result = PipelineResult(topic=topic)
    t_total = time.perf_counter()

    cb("Szukanie zrodel w DuckDuckGo", 0.1)
    hits = search_top(topic, n=n_sources)

    cb("Pobieranie tekstow", 0.25)
    docs: list[Document] = []
    per_source_progress = 0.4 / max(len(hits), 1)
    for i, hit in enumerate(hits):
        src = SourceResult(url=hit["url"], title=hit.get("title", ""))
        try:
            text = fetch_text(hit["url"])
            if len(text) > MAX_CHARS_PER_DOC:
                text = text[:MAX_CHARS_PER_DOC]
            src.chars = len(text)
            docs.append(Document(page_content=text, metadata={"source": hit["url"], "title": src.title}))
        except ScrapeError as e:
            src.error = str(e)
            result.errors_count += 1
            ERRORS.labels(kind="scrape").inc()
        result.sources.append(src)
        cb(f"Pobrano {i+1}/{len(hits)} zrodel", 0.25 + (i + 1) * per_source_progress)

    if not docs:
        raise RuntimeError("Nie udalo sie pobrac tekstu z zadnego zrodla.")

    result.original_chars = sum(s.chars for s in result.sources)
    ok_sources = [s for s in result.sources if not s.error]
    result.context_docs = [
        {"title": s.title or s.url, "url": s.url, "text": docs[i].page_content}
        for i, s in enumerate(ok_sources)
    ]

    cb("Generowanie podsumowania przez AI", 0.7)
    llm = get_llm()
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
    )
    t_llm = time.perf_counter()
    chain_out = chain.invoke({"input_documents": docs})
    result.llm_seconds = time.perf_counter() - t_llm
    LLM_SECONDS.observe(result.llm_seconds)
    result.summary = chain_out["output_text"].strip()
    result.summary_chars = len(result.summary)
    result.cache_hit = result.llm_seconds < CACHE_HIT_THRESHOLD_S
    CACHE_EVENTS.labels(
        layer="llm",
        result="hit" if result.cache_hit else "miss",
    ).inc()

    cb("Gotowe", 1.0)
    result.total_seconds = time.perf_counter() - t_total
    return result
