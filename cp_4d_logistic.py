from pathlib import Path
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from scipy.special import expit

# ===== ścieżki =====
DATASET_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\32x32x32_T100")
X_PATH = DATASET_DIR / "X.npy"
Y_PATH = DATASET_DIR / "y.npy"

# ===== parametry =====
MAX_SUBJECTS = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42

RANK = 2
N_ITER = 50
LR = 5e-2
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
    B_tensor = cp_to_tensor_4d(A, B, C, D)
    logits = np.einsum("nijkl,ijkl->n", X, B_tensor) + b0
    return logits


def fit_cp_logistic_simple(X, y, rank=2, n_iter=50, lr=5e-2, seed=123, l2=0.0):
    rng = np.random.default_rng(seed)

    n, p1, p2, p3, p4 = X.shape
    R = rank

    # większa skala inicjalizacji
    A = (rng.standard_normal((p1, R)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((p2, R)) * 0.1).astype(np.float32)
    C = (rng.standard_normal((p3, R)) * 0.1).astype(np.float32)
    D = (rng.standard_normal((p4, R)) * 0.1).astype(np.float32)

    # sensowny start dla interceptu
    p = float(y.mean())
    p = min(max(p, 1e-6), 1 - 1e-6)
    b0 = float(np.log(p / (1 - p)))

    history = []

    for it in range(n_iter):
        logits = predict_logits_cp(X, A, B, C, D, b0)
        probs = expit(logits)

        eps = 1e-8
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        reg = 0.5 * l2 * (
            np.sum(A ** 2) + np.sum(B ** 2) + np.sum(C ** 2) + np.sum(D ** 2)
        )
        total_loss = loss + reg
        history.append(float(total_loss))

        residual = probs - y  # shape (n,)
        grad_full = np.einsum("n,nijkl->ijkl", residual, X) / n  # shape (p1,p2,p3,p4)

        grad_A = np.zeros_like(A)
        grad_B = np.zeros_like(B)
        grad_C = np.zeros_like(C)
        grad_D = np.zeros_like(D)

        for r in range(R):
            a = A[:, r]
            b = B[:, r]
            c = C[:, r]
            d = D[:, r]

            grad_A[:, r] = np.einsum("ijkl,j,k,l->i", grad_full, b, c, d) + l2 * a
            grad_B[:, r] = np.einsum("ijkl,i,k,l->j", grad_full, a, c, d) + l2 * b
            grad_C[:, r] = np.einsum("ijkl,i,j,l->k", grad_full, a, b, d) + l2 * c
            grad_D[:, r] = np.einsum("ijkl,i,j,k->l", grad_full, a, b, c) + l2 * d

        grad_b0 = float(residual.mean())

        if it == 0 or it == n_iter - 1:
            print("\n=== NORMY GRADIENTÓW ===")
            print("||grad_A|| =", float(np.linalg.norm(grad_A)))
            print("||grad_B|| =", float(np.linalg.norm(grad_B)))
            print("||grad_C|| =", float(np.linalg.norm(grad_C)))
            print("||grad_D|| =", float(np.linalg.norm(grad_D)))
            print("grad_b0 =", float(grad_b0))

        A -= lr * grad_A
        B -= lr * grad_B
        C -= lr * grad_C
        D -= lr * grad_D
        b0 -= lr * grad_b0

        print(f"Iteracja {it+1}/{n_iter}, loss={total_loss:.6f}")

    return A, B, C, D, b0, history


def main():
    print("=== Wczytywanie datasetu ===")
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH)

    print("Pełny X shape:", X.shape)
    print("Pełny y shape:", y.shape)
    print("Pełne klasy:", Counter(y.tolist()))

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y), size=MAX_SUBJECTS, replace=False)

    X_small = np.asarray(X[idx], dtype=np.float32)
    y_small = y[idx]

    print("\n=== PODZBIÓR TESTOWY ===")
    print("X_small shape:", X_small.shape)
    print("y_small shape:", y_small.shape)
    print("Klasy w podzbiorze:", Counter(y_small.tolist()))

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

    print("\n=== TRENING CP-LOGISTIC ===")
    print("RANK =", RANK)
    print("N_ITER =", N_ITER)
    print("LR =", LR)

    A, B, C, D, b0, history = fit_cp_logistic_simple(
        X_train,
        y_train,
        rank=RANK,
        n_iter=N_ITER,
        lr=LR,
        seed=SEED,
        l2=0.0
    )

    print("\nHistoria loss:", history)

    print("\n=== DIAGNOSTYKA ===")
    print("loss start:", history[0])
    print("loss end:", history[-1])
    print("loss diff:", history[-1] - history[0])

    print("\n=== NORMY CZYNNIKÓW ===")
    for r in range(RANK):
        print(f"Komponent {r}:")
        print("||A[:,r]|| =", float(np.linalg.norm(A[:, r])))
        print("||B[:,r]|| =", float(np.linalg.norm(B[:, r])))
        print("||C[:,r]|| =", float(np.linalg.norm(C[:, r])))
        print("||D[:,r]|| =", float(np.linalg.norm(D[:, r])))

    print("b0 =", float(b0))

    logits_test = predict_logits_cp(X_test, A, B, C, D, b0)
    probs_test = expit(logits_test)
    y_pred = (probs_test >= 0.5).astype(int)

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