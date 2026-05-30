from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from scipy.special import expit

# =========================
# ŚCIEŻKI
# =========================
DATASET_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\61x73x61_T100_subjectwise")
INDEX_PATH = DATASET_DIR / "subject_index.csv"

# =========================
# PARAMETRY
# =========================
MAX_SUBJECTS = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42

RANK = 2
N_OUTER_ITER = 8
LR_BLOCK = 1e-2
SEED = 123


def cp_to_tensor_4d(A, B, C, D):
    p1, R = A.shape
    p2, _ = B.shape
    p3, _ = C.shape
    p4, _ = D.shape

    coef_tensor = np.zeros((p1, p2, p3, p4), dtype=np.float32)
    for r in range(R):
        coef_tensor += np.einsum("i,j,k,l->ijkl", A[:, r], B[:, r], C[:, r], D[:, r])
    return coef_tensor


def predict_logits_cp(X, A, B, C, D, b0):
    coef_tensor = cp_to_tensor_4d(A, B, C, D)
    logits = np.einsum("nijkl,ijkl->n", X, coef_tensor) + b0
    return logits


def logistic_loss_from_logits(logits, y):
    probs = expit(logits)
    eps = 1e-8
    return -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))


def init_factors(shape, rank, seed=123, scale=0.05):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((shape[0], rank)) * scale).astype(np.float32)
    B = (rng.standard_normal((shape[1], rank)) * scale).astype(np.float32)
    C = (rng.standard_normal((shape[2], rank)) * scale).astype(np.float32)
    D = (rng.standard_normal((shape[3], rank)) * scale).astype(np.float32)
    return A, B, C, D


def update_block_A(X, y, A, B, C, D, b0, lr):
    logits = predict_logits_cp(X, A, B, C, D, b0)
    probs = expit(logits)
    residual = probs - y
    grad_full = np.einsum("n,nijkl->ijkl", residual, X) / X.shape[0]

    grad_A = np.zeros_like(A)
    R = A.shape[1]
    for r in range(R):
        grad_A[:, r] = np.einsum("ijkl,j,k,l->i", grad_full, B[:, r], C[:, r], D[:, r])

    return A - lr * grad_A


def update_block_B(X, y, A, B, C, D, b0, lr):
    logits = predict_logits_cp(X, A, B, C, D, b0)
    probs = expit(logits)
    residual = probs - y
    grad_full = np.einsum("n,nijkl->ijkl", residual, X) / X.shape[0]

    grad_B = np.zeros_like(B)
    R = B.shape[1]
    for r in range(R):
        grad_B[:, r] = np.einsum("ijkl,i,k,l->j", grad_full, A[:, r], C[:, r], D[:, r])

    return B - lr * grad_B


def update_block_C(X, y, A, B, C, D, b0, lr):
    logits = predict_logits_cp(X, A, B, C, D, b0)
    probs = expit(logits)
    residual = probs - y
    grad_full = np.einsum("n,nijkl->ijkl", residual, X) / X.shape[0]

    grad_C = np.zeros_like(C)
    R = C.shape[1]
    for r in range(R):
        grad_C[:, r] = np.einsum("ijkl,i,j,l->k", grad_full, A[:, r], B[:, r], D[:, r])

    return C - lr * grad_C


def update_block_D(X, y, A, B, C, D, b0, lr):
    logits = predict_logits_cp(X, A, B, C, D, b0)
    probs = expit(logits)
    residual = probs - y
    grad_full = np.einsum("n,nijkl->ijkl", residual, X) / X.shape[0]

    grad_D = np.zeros_like(D)
    R = D.shape[1]
    for r in range(R):
        grad_D[:, r] = np.einsum("ijkl,i,j,k->l", grad_full, A[:, r], B[:, r], C[:, r])

    return D - lr * grad_D


def update_intercept(X, y, A, B, C, D, b0, lr):
    logits = predict_logits_cp(X, A, B, C, D, b0)
    probs = expit(logits)
    grad_b0 = float(np.mean(probs - y))
    return b0 - lr * grad_b0


def fit_cp_als_logistic(X, y, rank=2, n_outer_iter=8, lr_block=1e-2, seed=123):
    _, p1, p2, p3, p4 = X.shape
    A, B, C, D = init_factors((p1, p2, p3, p4), rank, seed=seed, scale=0.05)

    p = float(y.mean())
    p = min(max(p, 1e-6), 1 - 1e-6)
    b0 = float(np.log(p / (1 - p)))

    history = []

    for it in range(n_outer_iter):
        A = update_block_A(X, y, A, B, C, D, b0, lr_block)
        B = update_block_B(X, y, A, B, C, D, b0, lr_block)
        C = update_block_C(X, y, A, B, C, D, b0, lr_block)
        D = update_block_D(X, y, A, B, C, D, b0, lr_block)
        b0 = update_intercept(X, y, A, B, C, D, b0, lr_block)

        logits = predict_logits_cp(X, A, B, C, D, b0)
        loss = logistic_loss_from_logits(logits, y)
        history.append(float(loss))

        print(f"Epoka blokowa {it+1}/{n_outer_iter}, loss={loss:.6f}")

    return A, B, C, D, b0, history


def load_subject_tensor(npy_path: Path) -> np.ndarray:
    x = np.load(npy_path).astype(np.float32)
    return x


def main():
    print("=== FULL-RES CP-ALS LOGISTIC ===")
    print("Dataset:", DATASET_DIR)
    print("Index  :", INDEX_PATH)

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono: {INDEX_PATH}")

    df = pd.read_csv(INDEX_PATH)

    print("\nLiczba wszystkich subjectów w indexie:", len(df))
    print("Rozkład klas:", Counter(df["label_binary"].tolist()))

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(df), size=MAX_SUBJECTS, replace=False)
    df_small = df.iloc[idx].reset_index(drop=True)

    print(f"\n=== PODZBIÓR {MAX_SUBJECTS} ===")
    print("Rozkład klas:", Counter(df_small["label_binary"].tolist()))

    X_list = []
    y_list = []

    for i, row in df_small.iterrows():
        npy_path = Path(row["saved_npy"])
        y = int(row["label_binary"])

        x = load_subject_tensor(npy_path)
        X_list.append(x)
        y_list.append(y)

        if i == 0:
            print("Shape jednego subjecta:", x.shape)
            print("dtype:", x.dtype)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    print("\n=== TENSOR X ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Klasy:", Counter(y.tolist()))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n=== SPLIT ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train:", Counter(y_train.tolist()))
    print("y_test :", Counter(y_test.tolist()))

    print("\n=== TRENING CP-ALS LOGISTIC ===")
    print("RANK =", RANK)
    print("N_OUTER_ITER =", N_OUTER_ITER)
    print("LR_BLOCK =", LR_BLOCK)

    A, B, C, D, b0, history = fit_cp_als_logistic(
        X_train,
        y_train,
        rank=RANK,
        n_outer_iter=N_OUTER_ITER,
        lr_block=LR_BLOCK,
        seed=SEED
    )

    print("\nHistoria loss:", history)
    print("loss start:", history[0])
    print("loss end:", history[-1])
    print("loss diff:", history[-1] - history[0])

    logits_test = predict_logits_cp(X_test, A, B, C, D, b0)
    probs_test = expit(logits_test)
    y_pred = (probs_test >= 0.5).astype(int)

    print("\n=== DIAGNOSTYKA TEST ===")
    print("min logit:", float(logits_test.min()))
    print("max logit:", float(logits_test.max()))
    print("mean logit:", float(logits_test.mean()))

    print("min prob:", float(probs_test.min()))
    print("max prob:", float(probs_test.max()))
    print("mean prob:", float(probs_test.mean()))
    print("probs_test[:10]:", probs_test[:10])

    print("\n=== WYNIKI TEST ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, probs_test))
    except ValueError:
        print("ROC AUC: nie udało się policzyć")

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()