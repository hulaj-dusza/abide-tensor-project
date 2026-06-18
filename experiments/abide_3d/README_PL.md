# Klasyfikacja tensorowa ABIDE 3D

Ten katalog zawiera eksperymenty klasyfikacyjne oparte na tensorach 3D, wykonane na pochodnych danych spoczynkowego fMRI ze zbioru ABIDE Preprocessed.

Główny ukończony eksperyment opisany w tym module wykorzystuje mapy **ReHo** (*Regional Homogeneity*) do rozróżniania osób ze spektrum autyzmu (ASD) i osób z grupy kontrolnej.

W module znajduje się także osobny eksperyment fALFF, jednak szczegółowy opis poniżej dotyczy eksperymentu ReHo.

## Struktura katalogu

```text
abide_3d/
├── README.md
├── README_PL.md
│
├── docs/
│   └── legacy_abide_3d_results.md
│
├── reho/
│   ├── code/
│   │   ├── prepare_abide_reho_3d_subjectwise.py
│   │   └── cp_logistic_als_reho_3d_subjectwise.py
│   └── results/
│       ├── results_cp_logistic_als_reho_3d.csv
│       └── summary_cp_logistic_als_reho_3d.csv
│
└── falff/
    ├── code/
    │   ├── prepare_abide_falff_3d_subjectwise.py
    │   └── cp_logistic_als_falff_3d_subjectwise.py
    └── results/
        ├── results_cp_logistic_als_falff_3d.csv
        └── summary_cp_logistic_als_falff_3d.csv
```

## Eksperyment klasyfikacyjny ReHo

### Cel

Celem eksperymentu było sprawdzenie, czy niskorangowy model regresji logistycznej o strukturze tensorowej potrafi odróżnić osoby ze spektrum autyzmu od osób z grupy kontrolnej na podstawie przestrzennie uporządkowanych, trójwymiarowych map ReHo.

Zamiast spłaszczać każdą objętość mózgu do jednego długiego wektora cech, metoda zachowuje pierwotną strukturę przestrzenną 3D wokseli. Tensor współczynników modelu jest reprezentowany przez niskorangową dekompozycję CP.

### Dane

* Zbiór danych: **ABIDE Preprocessed**
* Pipeline przetwarzania: **CPAC**
* Pochodna funkcjonalna: **ReHo**
* Strategia przetwarzania: **filt_noglobal**
* Wymiary tensora wejściowego: **61 × 73 × 61**
* Liczba uczestników: **884**

  * ASD: **408**
  * grupa kontrolna: **476**

Dane źródłowe ABIDE oraz przygotowane tensory dla poszczególnych uczestników nie są umieszczone w repozytorium ze względu na ich rozmiar.

### Przygotowanie danych

Pipeline przygotowania map ReHo:

1. dopasowuje indywidualne mapy ReHo do danych fenotypowych;
2. przypisuje etykiety binarne: ASD = 1, grupa kontrolna = 0;
3. tworzy jeden tensor 3D dla każdego uczestnika;
4. wykonuje normalizację z-score osobno dla każdego uczestnika, wyłącznie dla wokseli niezerowych;
5. zapisuje przygotowane tensory i metadane potrzebne do klasyfikacji.

### Model

W eksperymencie zastosowano klasyfikator regresji logistycznej z tensorem współczynników o strukturze CP.

Tensor współczynników przybliżany jest jako:

[
\mathcal{B} \approx \sum_{r=1}^{R}
\mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r,
]

gdzie (R) oznacza rangę CP, natomiast (\circ) oznacza iloczyn zewnętrzny.

Parametry modelu są estymowane poprzez naprzemienną optymalizację macierzy czynnikowych CP oraz wyrazu wolnego regresji logistycznej.

## Procedura ewaluacji

Prezentowany wynik uzyskano dla konfiguracji:

* ranga CP: **9**
* metoda walidacji: **stratyfikowana 10-krotna walidacja krzyżowa**
* liczba cykli optymalizacji w każdym foldzie: **5**
* parametr regularyzacji logistycznej: **C = 0,05**
* ziarno losowości: **42**

Wyniki przedstawiają średnią skuteczność uzyskaną w 10 foldach walidacji.

## Wyniki

| Miara                          | Średnia | Odchylenie standardowe |
| ------------------------------ | ------: | ---------------------: |
| AUC                            |   0,612 |                  0,048 |
| Accuracy                       |   0,584 |                  0,037 |
| Czułość dla ASD                |   0,505 |                      — |
| Swoistość dla grupy kontrolnej |   0,651 |                      — |

Model uzyskał średni wynik AUC równy **0,612**, co wskazuje na zdolność rozróżniania osób z ASD i osób z grupy kontrolnej wyższą od poziomu losowego, ale nadal umiarkowaną.

Średnia accuracy wyniosła **0,584**. Swoistość była wyższa od czułości, co oznacza, że w tej konfiguracji model skuteczniej klasyfikował osoby z grupy kontrolnej niż osoby z ASD.

## Interpretacja

Wynik należy traktować jako eksploracyjny, a nie diagnostyczny.

ABIDE jest heterogenicznym, wieloośrodkowym zbiorem danych. Na skuteczność klasyfikacji mogą wpływać między innymi różnice między ośrodkami, protokoły akwizycji, wiek i płeć uczestników, ruch głowy, struktura próby, decyzje dotyczące preprocessingu, ranga modelu oraz siła regularyzacji.

Celem eksperymentu jest ocena, czy niskorangowe modelowanie tensorowe może wykorzystywać informacje zawarte w pochodnych neuroobrazowych 3D, zachowując ich strukturę przestrzenną.

## Odtwarzalność

Aby odtworzyć eksperyment ReHo:

1. pobierz mapy ReHo ze zbioru ABIDE Preprocessed, przetworzone pipeline’em CPAC ze strategią `filt_noglobal`;
2. przygotuj tensory dla uczestników za pomocą `reho/code/prepare_abide_reho_3d_subjectwise.py`;
3. w razie potrzeby zaktualizuj lokalne ścieżki do danych w skryptach;
4. uruchom `reho/code/cp_logistic_als_reho_3d_subjectwise.py`;
5. przeanalizuj pliki CSV zapisane w `reho/results/`.

Surowe pliki ABIDE i przygotowane tensory dla uczestników zostały celowo wyłączone z kontroli wersji.
