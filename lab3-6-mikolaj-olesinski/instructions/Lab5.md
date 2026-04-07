# Zadania Lista 5

Celem listy jest rozbudowa potoku przetwarzania i budowa modelu klasyfikacji wydźwięku emocjonalnego (sentiment).

**Wymagania**

- Wszystkie procedury należy zintegrować z potokiem dvc.
- Komendy należy uruchomić korzystając ze środowiska w kontenerze Docker. W przeciwnym razie przyznane będzie 80% punktów (zgodnie z zasadami oceniania).

**Wskazówki**
- Dokonując analizy modeli (zadania 2 i 3) wykonuj walidację krzyżową (cross-validation) na zbiorze uczącym.
- Uważaj na przeciek danych (data leakage) nie tylko podczas mierzenia skuteczności na zbiorze testowym, ale również w trakcie walidacji krzyżowej.
- Aby uniknąć przecieku danych (data leakage) warto korzystać z klasy Pipeline [(docs)](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) z biblioteki `sklearn`.
- Zanim cokolwiek dodasz do projeku zastanów się w którym stage'u powinno to być umieszczone. W wielu sytuacjach nie będzie jednego dobrego rozwiązania, należy wtedy postąpić według własnych preferencji zdając sobie sprawę z wad i zalet decyzji.


### Zadanie 1 (1 pkt/ 2 pkt) Śledzenie eksperymentów

Materiały do tego zadania znajdują się w Laboratorium 3: Narzędzia zarządzania eksperymentami.

W poprzedniej liście zadań zaimplementowano potok przetwarzania danych i uczenia modeli. W ramach tego zadania należy dodać do niego logowanie wyników.

Wybierz jedno z narzędzi: Weights & Biases (1 pkt) lub MLflow (2 pkt).

* Do skryptu przeznaczonego do ewaluacji modelu dodaj obsługę logowania wyników do wybranego narzędzia.
* Przekaż odpowiednie parametry i metryki.
* Dodaj wykres macierzy pomyłek dla zbioru uczącego i testowego.
* Załącz zrzuty ekranu z narzędzia pokazujące przebieg eksperymentów.

Wskazówka do MLflow: Poczytaj o networking'u w dockerze. Narzędzie Docker Compose ułatwia to zadanie.

### Zadanie 2 (3 pkt) Inżynieria cech (Feature engineering)

Bazując na przeprowadzonej EDA, dokonaj inżynierii cech w celu polepszenia wyników z poprzedniego zadania:
- użyj metod selekcji cech
- użyj metod redukcji wymiarowości
- (opcjonalnie) zaproponuj nowe cechy

Wykonaj kilka iteracji i eksperymentów, sprawdź, jakie kombinacje metod/cech pozwalają poprawić rezultaty, a jakie nie.

### Zadanie 3 (4 pkt) Wektoryzacja tekstu

a) (1 pkt) Używając biblioteki Spacy, dokonaj preprocessingu danych tekstowych. Usuń z tekstu elementy, które nie są nośnikami emocji. 

b) (3 pkt) Dokonaj wektoryzacji tak przetworzonego tekstu przy pomocy następujących metod:
- bag-of-words
- tf-idf
- word2vec (należy użyć pretrenowanego modelu)

Wyucz klasyfikator używając każdej z nich. Porównaj wyniki i spróbuj uzasadnić różnice w jakości.

### Zadanie 4 (1 pkt) Skuteczność na zbiorze testowym

Przygotuj krótki raport w Jupyter Notebook, w którym porównasz wyuczone modele w najlepszej dla nich konfiguracji na zbiorze testowym. Wyuczone mają być na całym zbiorze treningowym, czyli bez walidacji krzyżowej.  

Przeanalizuj uzyskane wyniki pod kątem dokonanej podczas ostatnich zajęć EDA. Czy zdobyta wiedza była przydatna? Czy wysnute wtedy wnioski znalazły potwierdzenie w wynikach modelu? 
