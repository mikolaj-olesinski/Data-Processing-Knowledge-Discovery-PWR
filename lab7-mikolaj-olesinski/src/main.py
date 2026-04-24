import streamlit as st

import charts
from cache import ping
from config import load_config
from history import clear_history, get_recent, load_history_df, save_query
from llm import chat_with_context
from metrics import start_metrics_server
from pipeline import PipelineResult, summarize_topic

start_metrics_server()

st.set_page_config(page_title="Lab7 — Web Summarizer", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      h1, h2, h3 {letter-spacing: -0.5px;}
      [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        background: #161925;
        border: 1px solid #262A38;
      }
      [data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
        background: #1F1D33;
        border-color: #3A3466;
      }
      [data-testid="stChatInput"] textarea {font-size: 0.95rem;}
      div[data-testid="stMetricValue"] {font-size: 1.4rem;}
      .hero-title {font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem;}
      .hero-sub {color: #A8A8B8; font-size: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.pipeline_result = None
    st.session_state.context_docs = []
    st.session_state.llm_times = []
    st.session_state.topic = None


def reset_session() -> None:
    st.session_state.messages = []
    st.session_state.pipeline_result = None
    st.session_state.context_docs = []
    st.session_state.llm_times = []
    st.session_state.topic = None


def load_from_history(entry: dict) -> None:
    result = PipelineResult.from_entry(entry)
    st.session_state.pipeline_result = result
    st.session_state.context_docs = result.context_docs
    st.session_state.topic = result.topic
    st.session_state.llm_times = [result.llm_seconds] if result.llm_seconds else []
    st.session_state.messages = [
        {"role": "user", "content": result.topic},
        {"role": "assistant", "content": result.summary},
    ]


try:
    cfg = load_config()
    config_ok = True
except RuntimeError as e:
    config_ok = False
    st.error(str(e))

redis_ok = ping()

with st.sidebar:
    st.markdown("# Web Summarizer")

    st.divider()
    st.markdown("**Status**")
    if config_ok:
        st.success(f"Gemini: {cfg.gemini_model}")
    else:
        st.error("GOOGLE_API_KEY: brak")
    if redis_ok:
        st.success("Redis: OK")
    else:
        st.error("Redis: brak polaczenia")

    st.divider()
    n_sources = st.slider(
        "Liczba zrodel (tylko nowy temat)",
        2, 5, 3,
        disabled=st.session_state.pipeline_result is not None,
    )

    if st.button("Nowy temat", use_container_width=True, type="primary"):
        reset_session()
        st.rerun()

    st.divider()
    st.markdown("**Historia (ostatnie 10)**")
    if redis_ok:
        recent = get_recent(10)
        if not recent:
            st.caption("Brak zapytan.")
        for idx, entry in enumerate(recent):
            label = entry["topic"][:44] + ("..." if len(entry["topic"]) > 44 else "")
            if st.button(
                f"{label}  ({entry['llm_seconds']:.1f}s)",
                key=f"resume_{entry.get('timestamp', idx)}",
                use_container_width=True,
            ):
                load_from_history(entry)
                st.rerun()
        if recent and st.button("Wyczysc historie", use_container_width=True):
            clear_history()
            st.rerun()
    else:
        st.caption("Redis niedostepny.")


tab_chat, tab_src, tab_stats = st.tabs(["Rozmowa", "Zrodla", "Statystyki"])


def _run_first_turn(topic: str) -> None:
    progress = st.progress(0.0, text="Start...")
    status_box = st.empty()

    def update(stage: str, pct: float) -> None:
        progress.progress(pct, text=stage)
        status_box.caption(f"{stage} ({int(pct * 100)}%)")

    try:
        result = summarize_topic(topic, n_sources=n_sources, progress_cb=update)
    except Exception as e:
        progress.empty()
        status_box.empty()
        st.error(f"Blad pipeline: {e}")
        return

    progress.empty()
    status_box.empty()

    st.session_state.pipeline_result = result
    st.session_state.context_docs = result.context_docs
    st.session_state.llm_times.append(result.llm_seconds)
    st.session_state.topic = topic
    save_query(result)

    st.markdown(result.summary)
    st.session_state.messages.append({"role": "assistant", "content": result.summary})

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Czas calk. [s]", f"{result.total_seconds:.1f}")
        c2.metric("Czas LLM [s]", f"{result.llm_seconds:.2f}")
        c3.metric("Zrodla OK/all", f"{len(result.sources) - result.errors_count}/{len(result.sources)}")
        c4.metric("Znaki oryg / str.", f"{result.original_chars} / {result.summary_chars}")
        if result.cache_hit:
            st.caption("Trafienie w Redis cache (LLM < 0.3 s)")


def _run_followup() -> None:
    with st.spinner("Mysle..."):
        try:
            ans, secs, _ = chat_with_context(
                st.session_state.messages,
                st.session_state.context_docs,
            )
        except Exception as e:
            st.error(f"Blad LLM: {e}")
            return
    st.session_state.llm_times.append(secs)
    st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})


with tab_chat:
    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown('<div class="hero-title">Co chcesz streszczac?</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="hero-sub">Wpisz temat w polu ponizej. Po wygenerowaniu podsumowania '
                'mozesz kontynuowac rozmowe, dopytywac i odwolywac sie do zrodel.</div>',
                unsafe_allow_html=True,
            )
            st.caption("np. *sztuczna inteligencja w medycynie*, *historia Wroclawia*, *kwantowe obliczenia*")
    else:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    placeholder = (
        "Podaj temat..." if not st.session_state.messages
        else "Dopytaj o szczegoly, rozwin punkt, porownaj zrodla..."
    )
    user_input = st.chat_input(
        placeholder,
        disabled=not (config_ok and redis_ok),
    )

    if user_input:
        topic_or_question = user_input.strip()
        if not topic_or_question:
            st.stop()

        st.session_state.messages.append({"role": "user", "content": topic_or_question})
        with st.chat_message("user"):
            st.markdown(topic_or_question)

        with st.chat_message("assistant"):
            if st.session_state.pipeline_result is None:
                _run_first_turn(topic_or_question)
            else:
                _run_followup()
        st.rerun()


with tab_src:
    result = st.session_state.pipeline_result
    if result is None:
        st.caption("Brak zrodel. Wpisz temat w zakladce *Rozmowa*, aby je pobrac.")
    else:
        st.markdown(f"**Temat:** {result.topic}")
        st.caption(f"Pobrane: {len(result.sources) - result.errors_count}/{len(result.sources)}")
        for i, s in enumerate(result.sources, 1):
            with st.container(border=True):
                if s.error:
                    st.markdown(f"**[{i}] {s.title or s.url}**")
                    st.caption(s.url)
                    st.error(f"Blad: {s.error}")
                else:
                    st.markdown(f"**[{i}] {s.title or s.url}**  —  {s.chars} zn.")
                    st.caption(s.url)


with tab_stats:
    if st.session_state.pipeline_result is None:
        st.caption("Wykresy pojawia sie po pierwszym zapytaniu.")
    else:
        result = st.session_state.pipeline_result
        df = load_history_df()

        col_a, col_b = st.columns(2)
        col_a.plotly_chart(charts.bar_original_vs_summary(result), use_container_width=True)
        col_b.plotly_chart(charts.line_llm_times(df), use_container_width=True)

        col_c, col_d = st.columns(2)
        col_c.plotly_chart(charts.bar_sources_per_query(df), use_container_width=True)
        col_d.plotly_chart(charts.pie_cache_hits(df), use_container_width=True)

        st.plotly_chart(
            charts.line_session_times(st.session_state.llm_times),
            use_container_width=True,
        )
