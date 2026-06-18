from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


# ============================================================
# KONFIGURACJA
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\HP\Searches\fmri-tensor-project-git")

DATA_DIR = PROJECT_ROOT / "61x73x61_falff_filt_noglobal_subjectwise"
INDEX_CSV = DATA_DIR / "subject_index_falff_filt_noglobal.csv"

RESULTS_DIR = PROJECT_ROOT / "results_cp_logistic_als_falff_3d"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SHAPE = (61, 73, 61)

# Najpierw porównujemy z najlepszym wariantem ReHo
MAX_SUBJECTS = None

RANKS = [6, 8, 9]
N_CYCLES = 5
N_SPLITS = 3
LOGREG_C = 0.3

LOGREG_MAX_ITER = 300
RANDOM_STATE = 42

# Standaryzacja każdego subjecta osobno na niezerowych wokselach
SUBJECT_ZSCORE = True

EPS = 1e-8


# ============================================================
# WCZYTYWANIE DANYCH
# ============================================================

def subjectwise_zscore(X: np.ndarray) -> np.ndarray:
    """
    Standaryzuje każdy obraz 3D osobno, tylko po niezerowych wokselach.

    X shape:
        (n_subjects, 61, 73, 61)
    """
    X_out = np.empty_like(X, dtype=np.float32)

    for i in range(X.shape[0]):
        vol = X[i].astype(np.float32, copy=True)

        mask = np.isfinite(vol) & (vol != 0)

        if mask.sum() > 10:
            mean = vol[mask].mean()
            std = vol[mask].std()

            if std > EPS:
                vol[mask] = (vol[mask] - mean) / std
                vol[~mask] = 0.0
            else:
                vol[:] = 0.0
        else:
            vol[:] = 0.0

        X_out[i] = vol

    return X_out


def load_falff_dataset(max_subjects=None):
    """
    Wczytuje subjectwise fALFF dataset:
        X: (n, 61, 73, 61)
        y: (n,)
        meta: DataFrame
    """
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Nie znaleziono indeksu: {INDEX_CSV}")

    meta = pd.read_csv(INDEX_CSV)

    if "npy_path" not in meta.columns:
        raise ValueError("Brakuje kolumny npy_path w indeksie.")

    if "y" not in meta.columns:
        raise ValueError("Brakuje kolumny y w indeksie.")

    meta = meta.copy()
    meta["y"] = meta["y"].astype(int)

    # Opcjonalne ograniczenie do mniejszej próbki testowej
    if max_subjects is not None:
        n_per_class = max_subjects // 2

        meta_asd = meta[meta["y"] == 1].sample(
            n=min(n_per_class, (meta["y"] == 1).sum()),
            random_state=RANDOM_STATE,
        )

        meta_control = meta[meta["y"] == 0].sample(
            n=min(n_per_class, (meta["y"] == 0).sum()),
            random_state=RANDOM_STATE,
        )

        meta = (
            pd.concat([meta_asd, meta_control], axis=0)
            .sample(frac=1.0, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    X_list = []

    for i, row in meta.iterrows():
        path = Path(row["npy_path"])

        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku .npy: {path}")

        vol = np.load(path).astype(np.float32)

        if vol.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Niepoprawny shape dla {path.name}: {vol.shape}, oczekiwano {EXPECTED_SHAPE}"
            )

        X_list.append(vol)

        if (i + 1) % 50 == 0:
            print(f"Wczytano {i + 1}/{len(meta)} subjectów...")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = meta["y"].to_numpy(dtype=np.int64)

    if SUBJECT_ZSCORE:
        print("Wykonuję subjectwise z-score...")
        X = subjectwise_zscore(X)

    return X, y, meta


# ============================================================
# FUNKCJE CP
# ============================================================

def normalize_cp_factors(A, B, C):
    """
    Normalizuje kolumny czynników CP i przenosi skalę do lambdas.

    A: (p1, R)
    B: (p2, R)
    C: (p3, R)
    """
    R = A.shape[1]
    lambdas = np.ones(R, dtype=np.float32)

    for r in range(R):
        norm_a = np.linalg.norm(A[:, r]) + EPS
        norm_b = np.linalg.norm(B[:, r]) + EPS
        norm_c = np.linalg.norm(C[:, r]) + EPS

        A[:, r] /= norm_a
        B[:, r] /= norm_b
        C[:, r] /= norm_c

        lambdas[r] *= norm_a * norm_b * norm_c

    return A, B, C, lambdas


def cp_linear_predictor(X, A, B, C, lambdas, bias):
    """
    Liczy predyktor liniowy:

        eta_i = bias + <X_i, B_cp>

    gdzie:

        B_cp = sum_r lambda_r a_r o b_r o c_r

    X shape:
        (n, p1, p2, p3)
    """
    comps = np.einsum(
        "nabc,ar,br,cr->nr",
        X,
        A,
        B,
        C,
        optimize=True,
    )

    eta = bias + comps @ lambdas

    return eta


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# MACIERZE DESIGN DLA ALS
# ============================================================

def make_design_for_A(X, B, C):
    """
    Dla aktualizacji A.

    Z_A shape:
        (n, p1 * R)
    """
    Z = np.einsum(
        "nabc,br,cr->nar",
        X,
        B,
        C,
        optimize=True,
    )

    return Z.reshape(X.shape[0], -1)


def make_design_for_B(X, A, C):
    """
    Dla aktualizacji B.

    Z_B shape:
        (n, p2 * R)
    """
    Z = np.einsum(
        "nabc,ar,cr->nbr",
        X,
        A,
        C,
        optimize=True,
    )

    return Z.reshape(X.shape[0], -1)


def make_design_for_C(X, A, B):
    """
    Dla aktualizacji C.

    Z_C shape:
        (n, p3 * R)
    """
    Z = np.einsum(
        "nabc,ar,br->ncr",
        X,
        A,
        B,
        optimize=True,
    )

    return Z.reshape(X.shape[0], -1)


def fit_logistic_block(Z, y):
    """
    Dopasowanie regresji logistycznej dla jednego bloku ALS.
    """
    clf = LogisticRegression(
        penalty="l2",
        C=LOGREG_C,
        solver="lbfgs",
        max_iter=LOGREG_MAX_ITER,
        fit_intercept=True,
        class_weight=None,
    )

    clf.fit(Z, y)

    coef = clf.coef_.ravel().astype(np.float32)
    bias = float(clf.intercept_[0])

    return coef, bias


# ============================================================
# TRENING CP-LOGISTIC ALS
# ============================================================

def fit_cp_logistic_als(X, y, rank, n_cycles, random_state):
    """
    Trenuje CP-logistic ALS dla danych 3D.
    """
    rng = np.random.default_rng(random_state)

    n, p1, p2, p3 = X.shape

    A = rng.normal(0.0, 1.0, size=(p1, rank)).astype(np.float32)
    B = rng.normal(0.0, 1.0, size=(p2, rank)).astype(np.float32)
    C = rng.normal(0.0, 1.0, size=(p3, rank)).astype(np.float32)

    A, B, C, lambdas = normalize_cp_factors(A, B, C)

    bias = 0.0

    for cycle in range(1, n_cycles + 1):
        print(f"    ALS cycle {cycle}/{n_cycles}")

        # -------------------------------
        # Aktualizacja A
        # -------------------------------
        Z_A = make_design_for_A(X, B, C)
        coef_A, bias = fit_logistic_block(Z_A, y)

        A = coef_A.reshape(p1, rank)

        A, B, C, lambdas = normalize_cp_factors(A, B, C)

        # -------------------------------
        # Aktualizacja B
        # -------------------------------
        Z_B = make_design_for_B(X, A, C)
        coef_B, bias = fit_logistic_block(Z_B, y)

        B = coef_B.reshape(p2, rank)

        A, B, C, lambdas = normalize_cp_factors(A, B, C)

        # -------------------------------
        # Aktualizacja C
        # -------------------------------
        Z_C = make_design_for_C(X, A, B)
        coef_C, bias = fit_logistic_block(Z_C, y)

        C = coef_C.reshape(p3, rank)

        A, B, C, lambdas = normalize_cp_factors(A, B, C)

        eta_train = cp_linear_predictor(X, A, B, C, lambdas, bias)
        prob_train = sigmoid(eta_train)

        try:
            auc_train = roc_auc_score(y, prob_train)
        except ValueError:
            auc_train = np.nan

        acc_train = accuracy_score(y, prob_train >= 0.5)

        print(f"        train AUC={auc_train:.4f}, train ACC={acc_train:.4f}")

    model = {
        "A": A,
        "B": B,
        "C": C,
        "lambdas": lambdas,
        "bias": bias,
        "rank": rank,
        "n_cycles": n_cycles,
        "logreg_C": LOGREG_C,
    }

    return model


def predict_proba_cp(X, model):
    eta = cp_linear_predictor(
        X,
        model["A"],
        model["B"],
        model["C"],
        model["lambdas"],
        model["bias"],
    )

    return sigmoid(eta)


# ============================================================
# METRYKI
# ============================================================

def evaluate_predictions(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    # Etykiety:
    # 1 = ASD
    # 0 = CONTROL
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    tp_asd = cm[0, 0]
    fn_asd = cm[0, 1]
    fp_asd = cm[1, 0]
    tn_control = cm[1, 1]

    sens_asd = tp_asd / (tp_asd + fn_asd + EPS)
    spec_control = tn_control / (tn_control + fp_asd + EPS)

    return {
        "auc": auc,
        "acc": acc,
        "sens_ASD": sens_asd,
        "spec_CONTROL": spec_control,
        "tp_ASD": tp_asd,
        "fn_ASD": fn_asd,
        "fp_ASD": fp_asd,
        "tn_CONTROL": tn_control,
        "pred_1": int(y_pred.sum()),
        "prob_mean": float(y_prob.mean()),
        "prob_std": float(y_prob.std()),
    }


def count_cp_params(shape, rank):
    p1, p2, p3 = shape
    return rank * (p1 + p2 + p3)


def count_dense_params(shape):
    p1, p2, p3 = shape
    return p1 * p2 * p3


# ============================================================
# EKSPERYMENT
# ============================================================

def run_experiment():
    print("=== CP-logistic ALS na ABIDE fALFF 3D ===")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"INDEX_CSV: {INDEX_CSV}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"EXPECTED_SHAPE: {EXPECTED_SHAPE}")
    print(f"MAX_SUBJECTS: {MAX_SUBJECTS}")
    print(f"RANKS: {RANKS}")
    print(f"N_SPLITS: {N_SPLITS}")
    print(f"N_CYCLES: {N_CYCLES}")
    print(f"LOGREG_C: {LOGREG_C}")
    print(f"SUBJECT_ZSCORE: {SUBJECT_ZSCORE}")
    print()

    X, y, meta = load_falff_dataset(max_subjects=MAX_SUBJECTS)

    print()
    print("Dane wczytane.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Rozkład klas:")
    print(pd.Series(y).value_counts().rename(index={0: "CONTROL", 1: "ASD"}))
    print()

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_rows = []

    dense_params = count_dense_params(EXPECTED_SHAPE)

    for rank in RANKS:
        print()
        print("=" * 70)
        print(f"RANK = {rank}")
        print("=" * 70)

        cp_params = count_cp_params(EXPECTED_SHAPE, rank)
        compression = dense_params / cp_params

        print(f"CP params: {cp_params}")
        print(f"Dense params: {dense_params}")
        print(f"Compression: {compression:.2f}x")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            print()
            print("-" * 70)
            print(f"Fold {fold}/{N_SPLITS}, rank={rank}")
            print("-" * 70)

            start_time = time.time()

            X_train = X[train_idx]
            y_train = y[train_idx]

            X_test = X[test_idx]
            y_test = y[test_idx]

            model = fit_cp_logistic_als(
                X_train,
                y_train,
                rank=rank,
                n_cycles=N_CYCLES,
                random_state=RANDOM_STATE + 1000 * rank + fold,
            )

            y_prob_test = predict_proba_cp(X_test, model)
            metrics = evaluate_predictions(y_test, y_prob_test)

            elapsed_sec = time.time() - start_time

            row = {
                "derivative": "falff",
                "pipeline": "cpac",
                "strategy": "filt_noglobal",
                "rank": rank,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "CP_params": cp_params,
                "dense_params": dense_params,
                "compression": compression,
                "N_CYCLES": N_CYCLES,
                "LOGREG_C": LOGREG_C,
                "SUBJECT_ZSCORE": SUBJECT_ZSCORE,
                "test_auc": metrics["auc"],
                "test_acc": metrics["acc"],
                "test_sens_ASD": metrics["sens_ASD"],
                "test_spec_CONTROL": metrics["spec_CONTROL"],
                "test_tp_ASD": metrics["tp_ASD"],
                "test_fn_ASD": metrics["fn_ASD"],
                "test_fp_ASD": metrics["fp_ASD"],
                "test_tn_CONTROL": metrics["tn_CONTROL"],
                "test_pred_1": metrics["pred_1"],
                "test_prob_mean": metrics["prob_mean"],
                "test_prob_std": metrics["prob_std"],
                "bias": model["bias"],
                "elapsed_sec": elapsed_sec,
            }

            all_rows.append(row)

            print()
            print("Wynik folda:")
            print(f"    AUC:          {metrics['auc']:.4f}")
            print(f"    ACC:          {metrics['acc']:.4f}")
            print(f"    Sens ASD:     {metrics['sens_ASD']:.4f}")
            print(f"    Spec CONTROL: {metrics['spec_CONTROL']:.4f}")
            print(f"    elapsed sec:  {elapsed_sec:.2f}")

            # Zapis cząstkowy po każdym foldzie
            results_df = pd.DataFrame(all_rows)
            results_path = RESULTS_DIR / "results_cp_logistic_als_falff_3d.csv"
            results_df.to_csv(results_path, index=False)

    results_df = pd.DataFrame(all_rows)

    results_path = RESULTS_DIR / "results_cp_logistic_als_falff_3d.csv"
    results_df.to_csv(results_path, index=False)

    summary = (
        results_df
        .groupby("rank")
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            test_sens_ASD_mean=("test_sens_ASD", "mean"),
            test_spec_CONTROL_mean=("test_spec_CONTROL", "mean"),
            elapsed_sec_mean=("elapsed_sec", "mean"),
        )
        .reset_index()
    )

    summary_path = RESULTS_DIR / "summary_cp_logistic_als_falff_3d.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print("=" * 70)
    print("ZAKOŃCZONO EKSPERYMENT fALFF 3D")
    print("=" * 70)
    print(f"Wyniki zapisano do: {results_path}")
    print(f"Podsumowanie zapisano do: {summary_path}")
    print()
    print("SUMMARY:")
    print(summary)


if __name__ == "__main__":
    run_experiment()