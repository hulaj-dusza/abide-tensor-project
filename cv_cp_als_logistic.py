from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from scipy.special import expit

# ===== ścieżki =====
DATASET_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\32x32x32_T100")
X_PATH = DATASET_DIR / "X.npy"
Y_PATH = DATASET_DIR / "y.npy"

# ===== parametry =====
MAX_SUBJECTS = 700
N_SPLITS = 5
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


def fit_cp_als_logistic(X, y, rank=3, n_outer_iter=10, lr_block=1e-2, seed=123):
    _, p1, p2, p3, p4 = X.shape
    A, B, C, D = init_factors((p1, p2, p3, p4), rank, seed=seed, scale=0.1)

    p = float(y.mean())
    p = min(max(p, 1e-6), 1 - 1e-6)
    b0 = float(np.log(p / (1 - p)))

    history = []

    for _ in range(n_outer_iter):
        A = update_block_A(X, y, A, B, C, D, b0, lr_block)
        B = update_block_B(X, y, A, B, C, D, b0, lr_block)
        C = update_block_C(X, y, A, B, C, D, b0, lr_block)
        D = update_block_D(X, y, A, B, C, D, b0, lr_block)
        b0 = update_intercept(X, y, A, B, C, D, b0, lr_block)

        logits = predict_logits_cp(X, A, B, C, D, b0)
        loss = logistic_loss_from_logits(logits, y)
        history.append(float(loss))

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

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    acc_scores = []
    auc_scores = []

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_small, y_small), start=1):
        print(f"\n=== FOLD {fold}/{N_SPLITS} ===")

        X_train = X_small[train_idx]
        X_test = X_small[test_idx]
        y_train = y_small[train_idx]
        y_test = y_small[test_idx]

        print("Train:", X_train.shape, Counter(y_train.tolist()))
        print("Test :", X_test.shape, Counter(y_test.tolist()))

        A, B, C, D, b0, history = fit_cp_als_logistic(
            X_train,
            y_train,
            rank=RANK,
            n_outer_iter=N_OUTER_ITER,
            lr_block=LR_BLOCK,
            seed=SEED + fold
        )

        print("loss start:", round(history[0], 6))
        print("loss end:", round(history[-1], 6))
        print("loss diff:", round(history[-1] - history[0], 6))

        logits_test = predict_logits_cp(X_test, A, B, C, D, b0)
        probs_test = expit(logits_test)
        y_pred = (probs_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, probs_test)

        acc_scores.append(acc)
        auc_scores.append(auc)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        print("Accuracy:", round(acc, 4))
        print("ROC AUC:", round(auc, 4))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    print("\n=== PODSUMOWANIE CV ===")
    print("Accuracy foldy:", np.round(acc_scores, 4))
    print("ROC AUC foldy:", np.round(auc_scores, 4))
    print("Średnie Accuracy:", round(float(np.mean(acc_scores)), 4))
    print("Std Accuracy:", round(float(np.std(acc_scores)), 4))
    print("Średnie ROC AUC:", round(float(np.mean(auc_scores)), 4))
    print("Std ROC AUC:", round(float(np.std(auc_scores)), 4))

    print("\n=== RAPORT ZBIORCZY ZE WSZYSTKICH FOLDÓW ===")
    print(confusion_matrix(all_y_true, all_y_pred))
    print(classification_report(all_y_true, all_y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()