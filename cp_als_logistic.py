from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from scipy.special import expit

# ===== ścieżki =====
DATASET_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\32x32x32_T100")
X_PATH = DATASET_DIR / "X.npy"
Y_PATH = DATASET_DIR / "y.npy"

# ===== parametry =====
MAX_SUBJECTS =  200
TEST_SIZE = 0.2
RANDOM_STATE = 42

RANK = 3
N_OUTER_ITER = 10
LR_BLOCK = 1e-2
SEED = 123


def cp_to_tensor_4d(A, B, C, D):
    p1, R = A.shape
    p2, _ = B.shape
    p3, _ = C.shape
    p4, _ = D.shape

    B_tensor = np.zeros((p1, p2, p3, p4), dtype=np.float32)
    for r in range(R):
        B_tensor += np.einsum("i,j,k,l->ijkl", A[:, r], B[:, r], C[:, r], D[:, r])
    return B_tensor


def predict_logits_cp(X, A, B, C, D, b0):
    coef_tensor = cp_to_tensor_4d(A, B, C, D)
    logits = np.einsum("nijkl,ijkl->n", X, coef_tensor) + b0
    return logits


def logistic_loss_from_logits(logits, y):
    probs = expit(logits)
    eps = 1e-8
    return -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))


def init_factors(shape, rank, seed=123, scale=0.1):
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

    A = A - lr * grad_A
    return A


def update_block_B(X, y, A, B, C, D, b0, lr):
    logits = predict_logits_cp(X, A, B, C, D, b0)
    probs = expit(logits)
    residual = probs - y
    grad_full = np.einsum("n,nijkl->ijkl", residual, X) / X.shape[0]

    grad_B = np.zeros_like(B)
    R = B.shape[1]
    for r in range(R):
        grad_B[:, r] = np.einsum("ijkl,i,k,l->j", grad_full, A[:, r], C[:, r], D[:, r])

    B = B - lr * grad_B
    return B


def update_block_C(X, y, A, B, C, D, b0, lr):
    logits = predict_logits_cp(X, A, B, C, D, b0)
    probs = expit(logits)
    residual = probs - y
    grad_full = np.einsum("n,nijkl->ijkl", residual, X) / X.shape[0]

    grad_C = np.zeros_like(C)
    R = C.shape[1]
    for r in range(R):
        grad_C[:, r] = np.einsum("ijkl,i,j,l->k", grad_full, A[:, r], B[:, r], D[:, r])

    C = C - lr * grad_C
    return C


def update_block_D(X, y, A, B, C, D, b0, lr):
    logits = predict_logits_cp(X, A, B, C, D, b0)
    probs = expit(logits)
    residual = probs - y
    grad_full = np.einsum("n,nijkl->ijkl", residual, X) / X.shape[0]

    grad_D = np.zeros_like(D)
    R = D.shape[1]
    for r in range(R):
        grad_D[:, r] = np.einsum("ijkl,i,j,k->l", grad_full, A[:, r], B[:, r], C[:, r])

    D = D - lr * grad_D
    return D


def update_intercept(X, y, A, B, C, D, b0, lr):
    logits = predict_logits_cp(X, A, B, C, D, b0)
    probs = expit(logits)
    grad_b0 = float(np.mean(probs - y))
    b0 = b0 - lr * grad_b0
    return b0


def fit_cp_als_logistic(X, y, rank=2, n_outer_iter=10, lr_block=1e-2, seed=123):
    _, p1, p2, p3, p4 = X.shape

    A, B, C, D = init_factors((p1, p2, p3, p4), rank, seed=seed, scale=0.1)

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


def main():
    print("=== WCZYTYWANIE DANYCH ===")
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH)

    print("Pełny X shape:", X.shape)
    print("Pełny y shape:", y.shape)
    print("Pełne klasy:", Counter(y.tolist()))

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y), size=MAX_SUBJECTS, replace=False)

    X_small = np.asarray(X[idx], dtype=np.float32)
    y_small = y[idx]

    print("\n=== PODZBIÓR ===")
    print("X_small shape:", X_small.shape)
    print("y_small shape:", y_small.shape)
    print("Klasy:", Counter(y_small.tolist()))

    X_train, X_test, y_train, y_test = train_test_split(
        X_small,
        y_small,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_small
    )

    print("\n=== SPLIT ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train:", Counter(y_train.tolist()))
    print("y_test:", Counter(y_test.tolist()))

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