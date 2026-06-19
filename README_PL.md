# Regresja tensorowa dla danych neuroobrazowych

[English version](README.md)

## Opis projektu

Repozytorium zawiera eksperymenty badawcze dotyczące regresji tensorowej oraz klasyfikacji danych neuroobrazowych z użyciem metod tensorowych.

Projekt bada, czy modele tensorowe o niskiej randze mogą identyfikować przestrzenne wzorce w danych obrazowych mózgu przy znacznie mniejszej liczbie parametrów niż klasyczne modele wokselowe.

Aktualne analizy dotyczą klasyfikacji osób z zaburzeniami ze spektrum autyzmu (ASD) i osób z grupy kontrolnej na podstawie danych neuroobrazowych pochodzących ze zbioru ABIDE.

## Cel badawczy

Głównym celem projektu jest ocena, czy wysokowymiarowe dane neuroobrazowe mogą być efektywnie reprezentowane i analizowane przy użyciu modeli uczenia maszynowego opartych na tensorach.

Projekt odpowiada między innymi na następujące pytania:

* Czy regresja tensorowa o niskiej randze pozwala rozróżniać uczestników z ASD i osoby z grupy kontrolnej?
* Jak modele CP wypadają w porównaniu z klasyczną regresją logistyczną opartą na wszystkich wokselach?
* Jaki wpływ na wyniki mają ranga tensora, siła regularyzacji oraz wybór preprocessingu?

## Dane

Projekt wykorzystuje dane pochodzące ze zbioru ABIDE Preprocessed.

Analizowane są różne reprezentacje danych neuroobrazowych, w tym:

* czterowymiarowe dane funkcjonalnego rezonansu magnetycznego,
* trójwymiarowe mapy Regional Homogeneity (ReHo),

Zadanie klasyfikacyjne ma postać:

```text
ASD = 1
CONTROL = 0
```

Surowe dane neuroobrazowe, pliki NIfTI oraz indywidualne tensory NumPy nie są przechowywane w tym repozytorium. Są one zapisywane lokalnie i wykluczone z repozytorium przez plik `.gitignore`.

## Metody

Głównym obszarem metodologicznym projektu jest regresja tensorowa, w szczególności regresja logistyczna oparta na rozkładzie CP.

Dla tensora opisującego uczestnika:

```math
\mathcal{X}_i \in \mathbb{R}^{p_1 \times p_2 \times p_3}
```

model przewiduje prawdopodobieństwo przynależności do grupy ASD za pomocą logistycznej regresji tensorowej:

```math
P(y_i = 1 \mid \mathcal{X}_i)
=
\sigma\left(
\beta_0 +
\left\langle \mathcal{X}_i, \mathcal{B} \right\rangle
\right)
```

gdzie $\mathcal{B}$ jest tensorem współczynników, $\beta_0$ jest wyrazem wolnym, a $\sigma$ oznacza funkcję sigmoidalną:

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

Aby ograniczyć liczbę parametrów, tensor współczynników jest aproksymowany przez rozkład CP:

```math
\mathcal{B}
\approx
\sum_{r=1}^{R}
\mathbf{a}_r \circ
\mathbf{b}_r \circ
\mathbf{c}_r
```

gdzie $R$ oznacza rangę CP, a $\mathbf{a}_r$, $\mathbf{b}_r$ i $\mathbf{c}_r$ są wektorami czynników odpowiadającymi kolejnym wymiarom tensora. W tej implementacji skala każdej składowej CP jest zawarta bezpośrednio w wektorach czynników, dlatego nie wprowadzano osobnych wag $\lambda_r$.


Projekt obejmuje eksperymenty z wykorzystaniem:

* CP-logistic regression,
* naprzemiennej blokowej optymalizacji współczynników,
* procedur inspirowanych CP-ALS,
* gradientowej optymalizacji modeli tensorowych,
* klasycznej regresji logistycznej woksel po wokselu jako modelu bazowego,
* stratyfikowanej walidacji krzyżowej,
* porównywania rang tensorowych i parametrów regularyzacji.

## Główne branche

Repozytorium jest podzielone na branche reprezentujące różne etapy rozwoju projektu oraz eksperymenty.

| Branch                          | Przeznaczenie                                            |
| ------------------------------- | -------------------------------------------------------- |
| `main`                          | Ogólny opis projektu i stabilna dokumentacja             |
| `feature/build-dataset`         | Budowa oraz przygotowanie zbioru danych                  |
| `feature/cp-4d-logistic`        | CP-logistic regression dla danych fMRI 4D                |
| `feature/cp-als-logistic`       | Eksperymenty CP-ALS-inspired logistic regression         |
| `feature/cv-comparison`         | Walidacja krzyżowa i porównywanie modeli                 |
| `feature/benchmark-flat-logreg` | Klasyczny model bazowy: regresja logistyczna wokselowa   |
| `feature/cp-3d-als-logistic`    | Eksperymenty klasyfikacyjne na mapach 3D ReHo            |

Lista branchy może być aktualizowana wraz z rozwojem projektu.

## Typowy przebieg pracy

1. Pobranie i lokalne przygotowanie danych neuroobrazowych.
2. Utworzenie tensorów dla uczestników oraz pliku indeksowego CSV.
3. Ustawienie lokalnych ścieżek i hiperparametrów w wybranym skrypcie.
4. Uruchomienie eksperymentu.
5. Zapis wyników dla foldów oraz wyników zbiorczych do plików CSV.
6. Porównanie rezultatów dla różnych rang, wariantów preprocessingu i modeli bazowych.

## Wymagania

Projekt wykorzystuje Python oraz biblioteki do obliczeń naukowych:

```text
numpy
pandas
scikit-learn
```

Część eksperymentów może wymagać dodatkowych bibliotek, zależnie od implementacji.

Podstawowe zależności można zainstalować poleceniem:

```bash
pip install numpy pandas scikit-learn
```

## Uwagi dotyczące odtwarzalności wyników

Skrypty zawierają lokalne ścieżki plików, które przed uruchomieniem na innym komputerze należy odpowiednio zmienić.

Dla odtwarzalności każdy eksperyment powinien opisywać:

* pochodną neuroobrazową i wariant preprocessingu,
* rozmiar tensora,
* kryteria wyboru uczestników,
* rodzaj modelu,
* rangę CP,
* parametr regularyzacji,
* ustawienia optymalizacji,
* schemat walidacji krzyżowej,
* ziarno losowości,
* użyte metryki jakości.

## Metryki oceny

W projekcie wykorzystywane są między innymi:

* ROC-AUC,
* accuracy,
* czułość dla ASD,
* swoistość dla grupy kontrolnej,
* balanced accuracy,
* czas działania.

Wyniki należy interpretować jako rezultaty eksploracyjne, a nie jako miarę jakości modelu diagnostycznego.

## Ograniczenia obecnej wersji

* Dobór hiperparametrów nie jest jeszcze w pełni realizowany wewnątrz nested cross-validation.
* W części eksperymentów nie zastosowano harmonizacji różnic między ośrodkami.
* Kowariaty demograficzne i kliniczne nie są obecnie uwzględniane we wszystkich modelach.
* Dane są heterogeniczne pod względem ośrodków badawczych oraz decyzji preprocessingowych.
* Wyniki powinny być porównywane między wieloma konfiguracjami przed formułowaniem końcowych wniosków.

## Dalsze kierunki pracy

* Systematyczne porównanie różnych rang CP i wartości regularyzacji.
* Uwzględnienie zmiennych współzależnych demograficznych, klinicznych i związanych z ruchem.
* Porównanie z dodatkowymi modelami uczenia maszynowego.

