import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pipeline import PipelineResult


def bar_original_vs_summary(result: PipelineResult) -> go.Figure:
    fig = go.Figure(data=[go.Bar(
        x=["Oryginal (suma zrodel)", "Streszczenie"],
        y=[result.original_chars, result.summary_chars],
        marker_color=["#4C78A8", "#F58518"],
        text=[f"{result.original_chars} zn.", f"{result.summary_chars} zn."],
        textposition="outside",
    )])
    ratio = result.summary_chars / result.original_chars if result.original_chars else 0
    fig.update_layout(
        title=f"Dlugosc: oryginal vs streszczenie (kompresja {ratio:.1%})",
        yaxis_title="Liczba znakow",
        showlegend=False,
        height=360,
    )
    return fig


def line_llm_times(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty("Czas odpowiedzi LLM", "Brak danych - wykonaj zapytanie")
    plot_df = df.reset_index(drop=True).copy()
    plot_df["nr"] = plot_df.index + 1
    fig = px.line(
        plot_df, x="nr", y="llm_seconds",
        markers=True, hover_data=["topic", "cache_hit"],
        title="Czas odpowiedzi LLM dla kolejnych zapytan [s]",
    )
    fig.update_layout(xaxis_title="Nr zapytania", yaxis_title="Czas LLM [s]", height=360)
    return fig


def bar_sources_per_query(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty("Zrodla per zapytanie", "Brak danych")
    plot_df = df.tail(15).reset_index(drop=True).copy()
    plot_df["label"] = plot_df["topic"].str.slice(0, 24) + (plot_df["topic"].str.len() > 24).map({True: "...", False: ""})
    plot_df["ok"] = plot_df["sources_count"] - plot_df["errors_count"]
    fig = go.Figure()
    fig.add_bar(name="OK", x=plot_df["label"], y=plot_df["ok"], marker_color="#54A24B")
    fig.add_bar(name="Bledy", x=plot_df["label"], y=plot_df["errors_count"], marker_color="#E45756")
    fig.update_layout(
        title="Zrodla per zapytanie (ostatnie 15)",
        barmode="stack",
        xaxis_title="Temat",
        yaxis_title="Liczba zrodel",
        height=360,
    )
    return fig


def pie_cache_hits(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty("Cache hits vs misses", "Brak danych")
    hits = int(df["cache_hit"].sum())
    misses = int(len(df) - hits)
    fig = go.Figure(data=[go.Pie(
        labels=["Cache hit", "Miss (LLM)"],
        values=[hits, misses],
        hole=0.45,
        marker=dict(colors=["#54A24B", "#4C78A8"]),
    )])
    fig.update_layout(title=f"Cache hits vs misses ({hits}/{hits+misses})", height=360)
    return fig


def line_session_times(times: list[float]) -> go.Figure:
    if not times:
        return _empty("Czasy LLM w biezacej sesji", "Brak tur rozmowy")
    xs = list(range(1, len(times) + 1))
    fig = go.Figure(data=[go.Scatter(
        x=xs, y=times, mode="lines+markers",
        line=dict(color="#8B7FD7", width=2),
        marker=dict(size=8, color="#F58518"),
    )])
    fig.update_layout(
        title="Czasy LLM w biezacej sesji (per tura) [s]",
        xaxis_title="Nr tury",
        yaxis_title="Czas LLM [s]",
        height=360,
    )
    return fig


def _empty(title: str, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14))
    fig.update_layout(title=title, height=360, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig
