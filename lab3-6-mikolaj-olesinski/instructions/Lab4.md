# Zadania Lista 4: Analiza eksploracyjna danych

**Wymagania**

- Komendy należy uruchomić korzystając ze środowiska w kontenerze Docker. W przeciwnym razie przyznane będzie 80% punktów (zgodnie z zasadami oceniania).
- Wszystkie procedury należy zintegrować z potokiem dvc.

**Wskazówki**

- Dokonując analizy modeli wykonuj walidację krzyżową (cross-validation) na zbiorze uczącym. Uważaj na przeciek wiedzy / danych (knowledge / data leakage).
- Aby uniknąć przypadkowego przecieku danych warto korzystać z klasy Pipeline [(docs)](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) z biblioteki `sklearn`.

**Uwaga**
Jeśli przetwarzanie danych jest zbyt długie lub wymaga zbyt dużo pamięci, można:

- ograniczyć zbiór danych "na stałe" czyli we wstępnym etapie DVC.
- ograniczyć zbiór danych tymczasowo na potrzeby danej analizy.
  W obu przypadkach proszę pamiętać - próbka musi być reprezentatywna.

### Zadanie 1 (1 pkt) Dodanie cech

Skonstruuj co najmniej 3 dodatkowe cechy na podstawie danych oraz wiedzy dziedzinowej i umieść je jako osobne kolumny. Przykładem takiej cechy może być długość tekstu.
Konstrukcje umieść w odpowiednim etapie DVC.

### Zadanie 2 (5 pkt) Analiza EDA

Celem zadania jest dokonanie analizy eksploracyjnej (EDA) zadanego zbioru danych, pod kątem zastosowania do treningu modelu sentymentu, przy pomocy poznanych narzędzi.

W ramach zadania analizujemy wszystkie cechy zawarte w zbiorze *oraz te utworzone w zadaniu 1*.

Na podstawie danych numerycznych/kategorycznych postaraj się odpowiedzieć na następujące pytania:

- ile znajduje się w zbiorze cech kategorycznych, a ile numerycznych?
- czy zmienna wyjściowa jest kategoryczna, czy numeryczna?
- czy i ile w zbiorze jest brakujących wartości? Dla jakich zmiennych? Co z tego wynika? Jakie są możliwe sposoby radzenia sobie z brakującymi wartościami?
- czy któreś z cech są skorelowane? Co z tego może wynikać?
- czy któraś z cech koreluje ze zmienną wyjściową? Jeśli tak - która? Czy któraś nie koreluje?
- czy zbiór danych wydaje się być wystarczająco informacyjny by rozwiązać zadanie analizy sentymentu?

Zbadaj również dane tekstowe:

- czy któreś ze słów wydają się dominować w zbiorze?
- czy najpopularniejsze słowa różnią się znacząco pomiędzy klasami? Czy potrafisz wyróżnić słowa mogące wpływać w znaczym stopniu na sentyment?
- jaka jest charakterystyka tekstu (np. długość, czystość)?

Nie ograniczaj się jedynie do odpowiedzi na powyższe pytania.

- Zdefiniuj i odpowiedz na dodatkowe 2 pytania które uznasz za istotne w kontekście zadania.

Opisz swoje ciekawe spostrzeżenia co do danych. Spróbuj na podstawie uzyskanych informacji zarekomendować dalsze działania w zakresie posiadanej wiedzy.

Rezultatem pracy niech będzie prosty, krótki raport podsumowujący wiedzę, którą udało się uzyskać z EDA. Powinien zawierać sedno analiz i najważniejsze wnioski które udało się uzyskać. Raport powinien być w formacie Jupyter Notebook'a.

Co do formuły raportu - załóż, że pracujesz w dziale R&D renomowanej firmy i musisz przygotować krótkie streszczenie wiedzy dla swojego przełożonego, by mógł on podjąć decyzje o dalszych etapach prac. **Krótko, treściwie, same konkrety, wnioski i rekomendacje** :)

### Zadanie 3 (2 pkt) Czyszczenie danych i przetwarzanie wstępne

a) **(0.5 pkt)** Rozwiąż kwestię brakujących wartości - bazując na wykonanej EDA, uzupełnij brakujące wartości, usuń przypadki z brakującymi wartościami lub usuń kolumny zawierające brakujące wartości.

b) **(0.5 pkt)** Zakoduj zmienne kategoryczne i przeskaluj zmienne numeryczne

c) **(0.5 pkt)** Dokonaj czyszczenia danych tekstowych (należy zaproponować metodę czyszczenia). Można użyć do tego np. biblioteki Spacy, regex'y.

d) **(0.5 pkt)** Dokonaj wektoryzacji danych tekstowych metodą BoW

### Zadanie 4 (2 pkt) Uczenie modeli

Wytrenuj i porównaj dwa klasyczne algorytmy ML - SVM i Random Forest oraz klasyfikator Dummy używany w trakcie listy 3.

Przeprowadź następujące eksperymenty:

- użyj tylko danych tekstowych
- użyj pozostałych danych oprócz tekstowych
- użyj wszystkich danych

Zaraportuj metryki klasyfikacji stosując walidację krzyżową (cross-validation). Nie korzystamy ze zbioru testowego.

### Zadanie 5 (dodatkowe, 1 pkt) Przechowywanie notatników

Porównywanie notebooków za pomocą `git` jest problematyczne, dlatego najlepiej przechowywać je z wyczyszczonym outputem a wykonane przechowywać za pomocą dvc w osobnej lokalizacji. Dodatkową zaletą jest redukcja rozmiaru repozytorium gdy notebook'i zawierają dużo obrazków.

Dodaj do DVC stage, który będzie wykonywał napisany wyczyszczony notebook zapisując w folderze `data` lub jego podfolderze, dodaj odpowiednie zależności. Na potrzeby laboratorium dodaj `cache:false` do wyjściowego notebooka, aby był trakowany przez git. Wykorzystaj komendę `jupyter nbconvert` lub narzędzie [papermill](https://papermill.readthedocs.io/en/latest/).
