import time
from functools import lru_cache

from langchain.globals import set_llm_cache
from langchain_community.cache import RedisCache
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from cache import get_redis
from config import load_config

_cache_initialized = False


def _init_cache_once() -> None:
    global _cache_initialized
    if not _cache_initialized:
        set_llm_cache(RedisCache(redis_=get_redis()))
        _cache_initialized = True


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    cfg = load_config()
    _init_cache_once()
    return ChatGoogleGenerativeAI(
        model=cfg.gemini_model,
        google_api_key=cfg.google_api_key,
        temperature=0.3,
    )


SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Jestes asystentem tworzacym zwiezle, spojne podsumowania w jezyku polskim. "
               "Skup sie na kluczowych faktach, unikaj powtorzen, pisz klarownym jezykiem."),
    ("human", "Streszc ponizszy tekst w 5-8 zdaniach:\n\n{text}"),
])


def summarize_text(text: str) -> tuple[str, dict]:
    llm = get_llm()
    chain = SUMMARIZE_PROMPT | llm
    response = chain.invoke({"text": text})
    usage = getattr(response, "usage_metadata", None) or {}
    return response.content, dict(usage)


CHAT_SYSTEM = (
    "Jestes asystentem odpowiadajacym na pytania na bazie dostarczonych zrodel. "
    "Odpowiadaj po polsku, konkretnie i rzeczowo. Gdy to pomocne, cytuj zrodla numerem [1], [2], ... "
    "Jesli pytanie wychodzi poza zakres zrodel, wyraznie to zaznacz zanim odpowiesz z wiedzy ogolnej."
)

MAX_CONTEXT_CHARS = 40_000


def _build_context_block(docs: list[dict]) -> str:
    if not docs:
        return "(brak zrodel)"
    parts: list[str] = []
    budget = MAX_CONTEXT_CHARS
    per_doc = max(budget // max(len(docs), 1), 2000)
    for i, d in enumerate(docs, 1):
        chunk = d.get("text", "")[:per_doc]
        title = d.get("title") or d.get("url") or f"zrodlo {i}"
        url = d.get("url", "")
        parts.append(f"[Zrodlo {i}: {title} - {url}]\n{chunk}")
    return "\n\n---\n\n".join(parts)


def chat_with_context(messages: list[dict], docs: list[dict]) -> tuple[str, float, dict]:
    """Follow-up chat using already-scraped docs as context.

    messages: [{"role": "user"|"assistant", "content": str}, ...]
    docs: [{"title", "url", "text"}, ...]
    Returns: (answer, seconds, usage_metadata).
    """
    llm = get_llm()
    lc_msgs: list = [
        SystemMessage(content=CHAT_SYSTEM),
        HumanMessage(content=f"Kontekst ze zrodel:\n\n{_build_context_block(docs)}"),
        AIMessage(content="Rozumiem. Mam dostep do tych zrodel i odpowiem na kolejne pytania."),
    ]
    for m in messages:
        if m["role"] == "user":
            lc_msgs.append(HumanMessage(content=m["content"]))
        else:
            lc_msgs.append(AIMessage(content=m["content"]))

    t = time.perf_counter()
    resp = llm.invoke(lc_msgs)
    secs = time.perf_counter() - t
    usage = getattr(resp, "usage_metadata", None) or {}
    return resp.content, secs, dict(usage)
