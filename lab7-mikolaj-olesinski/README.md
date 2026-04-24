# Zadania Lista 7

**Uwagi ogólne**

1. Z powodu, że przyszłe zajęcia będą skrócone, może braknąć czasu na indywidualną prezentację rozwiązań. 
Proszę więc przyłożyć się do tworzenia zrzutów ekranu.  

## (do 6 pkt) 1. Zbuduj frontend aplikacji

W tym zadaniu stworzysz aplikację opartą na bibliotece Streamlit, która automatycznie pobiera treści z sieci i generuje ich podsumowanie przy użyciu dużego modelu językowego (LLM).

- LLM: Możesz skorzystać z darmowych API chmurowych (np. Google Gemini API - model gemini-2.5-flash-lite, gemini-1.5-flash, Groq API, ...) lub uruchomić model lokalnie wykorzystując narzędzie Ollama (np. modele llama3, mistral, qwen).
- Scrapowanie tekstu: BeautifulSoup4, trafilatura, newspaper3k lub biblioteka wikipedia.
- (Dla wersji za 5 pkt) Wyszukiwanie: duckduckgo-search (darmowa biblioteka w Pythonie), Google Custom Search API, Tavily API, ...

Warianty zadania (wybierz jeden):

- Wersja podstawowa (4 pkt): Aplikacja pozwala użytkownikowi na podanie bezpośredniego linku do artykułu na Wikipedii. Skrypt pobiera treść artykułu, oczyszcza ją z tagów HTML, a następnie wysyła do modelu LLM z promptem proszącym o wygenerowanie zwięzłego podsumowania.

- Wersja rozszerzona (6 pkt): Aplikacja pozwala użytkownikowi na wpisanie dowolnego tematu. Skrypt w tle wykonuje przeszukiwanie sieci (np. przez DuckDuckGo), pobiera 2-3 najwyżej pozycjonowane artykuły, ekstrahuje z nich tekst, łączy go i prosi LLM o wygenerowanie spójnego podsumowania na bazie znalezionych źródeł.

Wymagane funkcjonalności aplikacji (niezależnie od wariantu):

- input pozwalający na podanie linku (wer. podst.) lub tematu (wer. rozsz.)

- użycie metody cache'ującej wyniki scrapowania oraz odpowiedzi z LLM, by zaoszczędzić transfer, limit zapytań API i czas

- wyświetlenie wygenerowanego przez LLM podsumowania

- wykres słupkowy porównujący długość oryginalnego tekstu (w znakach lub słowach) z długością wygenerowanego podsumowania

- wykres przedstawiający czas odpowiedzi modelu LLM (w sekundach) dla kolejnych zapytań w danej sesji

- dwa dowolne inne wykresy przedstawiające statystyki  potencjalnie istotne dla użytkownika (np. liczba zebranych źródeł, liczba błędów pobierania, szacowana liczba użytych tokenów, jeśli API to zwraca)

- progressbar pokazujący postęp pracy (np. Szukanie -> Pobieranie tekstów -> Generowanie odpowiedzi przez AI)

Stworzoną aplikację udokumentuj zrzutami ekranu.


## (2 pkt) 2. Zbuduj obraz Dockerowy aplikacji
Na bazie kodu z pkt 1 stwórz obraz Dockera zawierający i uruchamiający aplikację, umożliwiający
dostanie się do niej z komputera hosta.
Obraz ma mieć charakter produkcyjny, czyli:

- Obraz ma być minimalny, zawierać tylko niezbędne biblioteki i pliki.

- Obraz ma być kompletny i nie wymagać podmontowania żadnych zewnętrznych plików.

- Obraz ma nie zawierać żadnych danych autoryzacyjnych, które są potrzebne do uruchomienia aplikacji, w szczególności kluczy API do modeli LLM (możesz np. użyć zmiennych środowiskowych).


## (2 pkt) 3. Zbieraj statystyki produkcyjne
Używając Docker Compose, stwórz środowisko umożliwiające zbieranie statystyk:

- złącz swoją aplikację z obrazem Grafany w jedno środowisko dockerowe

- rozpocznij zbieranie trzech różnych statystyk z aplikacji. Przykładowe statystyki: ilość uruchomień aplikacji, ilość zapytań obsłużonych z cache, czas pobierania danych z sieci, czas oczekiwania na odpowiedź LLM (timing), całkowita liczba przetworzonych znaków/słów, ilość błędów (np. timeouty API, błędy 403 przy scrapowaniu).

- stwórz dashboard w Grafanie zawierający wykresy stworzone na bazie zbieranych metryk

- załóż alert na któryś z wykresów Grafany. Udokumentuj zrzutem ekranu.

Stworzone wykresy udokumentuj zrzutami ekranu.
