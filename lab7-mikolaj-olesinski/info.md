# Lab7 — Web Summarizer

Aplikacja do generowania podsumowań dowolnego tematu na bazie artykułów wyszukanych w sieci.

## Uruchomienie

```bash
cp .env.example .env
# wpisac GOOGLE_API_KEY do .env
docker compose up -d --build
```

Adresy:

- Aplikacja: http://localhost:8501
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (wejście bez logowania jako anonymous viewer)

## Co jest zrobione

### Zadanie 1 — Frontend

- Input tematu (chat), slider liczby źródeł 2–5
- Wyszukiwanie w sieci przez DuckDuckGo (biblioteka `ddgs`) z fallbackiem na Google/Bing backend
- Scraping artykułów biblioteką `trafilatura`
- Podsumowanie generowane przez Gemini 2.5-flash-lite (map-reduce: mapa faktów → spójne streszczenie)
- Cache wyników scrapowania (7 dni), wyszukiwania (1 dzień) i odpowiedzi LLM w Redis
- Progressbar: *Szukanie źródeł → Pobieranie tekstów → Generowanie AI → Gotowe*
- Zakładki: **Rozmowa** (chat + follow-up), **Źródła** (lista pobranych linków), **Statystyki** (5 wykresów)
- Historia zapytań w sidebarze (ostatnie 10, zapisane w Redis)

### Zadanie 2 — Docker

- Brak danych autoryzacyjnych w obrazie — `GOOGLE_API_KEY` przekazywany przez zmienną środowiskową
- Obraz kompletny, nie wymaga montowania plików zewnętrznych
- `.env.example` jako szablon (bez sekretów)

### Zadanie 3 — Prometheus + Grafana

- Docker Compose łączy aplikację z Prometheusem i Grafaną
- 3 metryki eksportowane z aplikacji
- Dashboard w Grafanie z 3 panelami (auto-provisioning)
- Alert na wzroście błędów

## Architektura

```
user → Streamlit (8501) → pipeline: search → scrape → LLM
                              ↓
                           Redis (cache + historia)
                              ↓
                           Prometheus (9100 exporter) → pull /metrics (9090) → Grafana (3000)
```

Moduły w `src/`:

- `main.py` — UI Streamlit, 3 zakładki, historia
- `pipeline.py` — orkiestracja search → scrape → LLM, pomiary czasu, zliczanie błędów, cache hit
- `search.py` — DuckDuckGo z retry + backoff
- `scraper.py` — trafilatura, skrócenie do 60k znaków
- `llm.py` — Gemini przez LangChain + `RedisCache`
- `cache.py` — dekorator `@redis_cache` z TTL i haszowaniem kluczy
- `charts.py` — 5 wykresów Plotly
- `metrics.py` — eksporter Prometheusa (port 9100)
- `history.py` — zapis zapytań w Redis
- `config.py` — wczytanie `.env`

## Dlaczego LangChain

LangChain jest użyty w `src/llm.py` po to, żeby nie pisać samemu kilku powtarzalnych kawałków:

- **Jednolite API do Gemini** (`ChatGoogleGenerativeAI`) — zmiana modelu na inny wymaga podmiany jednej linii, nie całej warstwy HTTP.
- **Map-reduce dla długich tekstów** — artykuły po połączeniu potrafią mieć 100–200 tys. znaków, co nie zmieści się w jednym zapytaniu. Chain najpierw robi „mapę faktów" z każdego kawałka, potem łączy wszystko w spójne streszczenie.
- **`RedisCache`** — wbudowane cache'owanie odpowiedzi LLM: identyczny prompt + model = natychmiastowa odpowiedź z Redis (widać po `Trafienie w Redis cache (LLM < 0.3 s)` w UI). Bez LangChaina trzeba by pisać własne haszowanie promptu i klucze Redis.
- **Format wiadomości chat** — follow-up w zakładce *Rozmowa* korzysta z historii wiadomości (user/assistant) w standardzie LangChain, więc LLM widzi cały kontekst dialogu, nie tylko ostatnie pytanie.

## Stan aplikacji i historia

Aplikacja trzyma dwa rodzaje stanu:

- **Stan sesji (`st.session_state`)** — pamiętany tylko w zakładce przeglądarki: aktualny temat, kontekst artykułów, wiadomości chatu, czasy LLM w sesji. Przycisk *Nowy temat* czyści sesję.
- **Stan trwały (Redis)** — przetrwa restart aplikacji:
  - **Cache** scrape (7 dni), search (1 dzień), LLM (bezterminowo) — `src/cache.py`
  - **Historia zapytań** (ostatnie 500, każdy wpis zawiera temat, streszczenie, źródła, czas LLM, czy był cache hit) — `src/history.py`. W sidebarze widać ostatnie 10; kliknięcie wpisu przywraca pełny wynik bez nowego zapytania do API.
  - Klucze: `lab7:scrape:<hash>`, `lab7:search:<hash>`, `lab7:history` (list), LangChain trzyma swoje cache pod własnymi kluczami.

Dzięki temu ten sam temat drugi raz nie kosztuje ani zapytania do DuckDuckGo, ani scrapowania, ani tokenów Gemini — wszystko wraca z Redis w <0.3 s.

## Metryki Prometheusa

1. `lab7_llm_seconds` — histogram czasu odpowiedzi Gemini (buckets 0.1–60s)
2. `lab7_cache_events_total` — counter trafień/pudeł cache (labels: `layer=scrape|search|llm`, `result=hit|miss`)
3. `lab7_errors_total` — counter błędów pipeline (labels: `kind=scrape|search`)

## Panele Grafany (dashboard „lab7")

1. **Czas LLM p50/p95** — `histogram_quantile` z `lab7_llm_seconds_bucket`
2. **Cache rate/min** — `rate(lab7_cache_events_total[1m])` rozbity po `layer` i `result`
3. **Błędy rate/min** — `rate(lab7_errors_total[1m])` rozbity po `kind`

**Alert**: *Wysoki wskaźnik błędów* — zapala się, gdy `rate(lab7_errors_total[1m]) > 0.05` utrzymuje się przez 2 minuty.

## Wykresy w aplikacji (zakładka Statystyki)

1. **Bar** — długość oryginalnych tekstów vs długość podsumowania (znaki) — *wymagany przez README*
2. **Line** — czasy odpowiedzi LLM dla kolejnych zapytań w historii — *wymagany przez README*
3. **Bar** — liczba źródeł OK / błędnych na zapytanie (ostatnie 15)
4. **Pie** — procent trafień cache vs zapytań do LLM
5. **Line** — czasy LLM w bieżącej sesji (per tura rozmowy)
