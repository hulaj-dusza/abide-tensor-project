from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

# =========================
# ŚCIEŻKI
# =========================
RAW_DATA_DIR = Path(r"C:\Users\HP\Searches\abide_data")
FMRI_DIR = RAW_DATA_DIR / "ABIDE_pcp" / "cpac" / "filt_global"
PHENO_PATH = RAW_DATA_DIR / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"

OUT_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\32x32x32_T150")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# PARAMETRY DATASETU
# =========================
TARGET_SHAPE = (32, 32, 32)
TARGET_T = 150

# =========================
# FUNKCJE POMOCNICZE
# =========================
def resize_3d_volume(volume_3d: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    zoom_factors = [t / s for t, s in zip(target_shape, volume_3d.shape)]
    resized = zoom(volume_3d, zoom=zoom_factors, order=1)
    return resized.astype(np.float32)

def standardize_subject(data_4d: np.ndarray) -> np.ndarray:
    mean = data_4d.mean()
    std = data_4d.std()
    if std < 1e-8:
        return (data_4d - mean).astype(np.float32)
    return ((data_4d - mean) / std).astype(np.float32)

def file_id_from_name(filename: str) -> str:
    return filename.split("_func")[0]

# =========================
# WCZYTANIE FENO
# =========================
if not PHENO_PATH.exists():
    raise FileNotFoundError(f"Nie znaleziono pliku fenotypowego: {PHENO_PATH}")

df = pd.read_csv(PHENO_PATH)

required_cols = {"FILE_ID", "DX_GROUP"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Brakuje kolumn w CSV: {missing}")

df["FILE_ID"] = df["FILE_ID"].astype(str)
dx_map = dict(zip(df["FILE_ID"], df["DX_GROUP"]))

# =========================
# LISTA PLIKÓW
# =========================
all_files = sorted(FMRI_DIR.glob("*.nii.gz"))
print("Liczba wszystkich plików .nii.gz:", len(all_files))

eligible_files = []
eligible_labels = []

for f in all_files:
    file_id = file_id_from_name(f.name)

    if file_id not in dx_map:
        continue

    img = nib.load(f)
    shape = img.header.get_data_shape()

    if len(shape) != 4:
        continue

    t = shape[3]
    if t < TARGET_T:
        continue

    dx = int(dx_map[file_id])
    if dx not in (1, 2):
        continue

    # ASD=1 -> 1, Control=2 -> 0
    y = 1 if dx == 1 else 0

    eligible_files.append(f)
    eligible_labels.append(y)

print(f"Liczba subjectów z T >= {TARGET_T}:", len(eligible_files))
print("Rozkład klas:", Counter(eligible_labels))

# =========================
# BUDOWA X i y
# =========================
X_list = []
y_list = []
subject_rows = []
selected_paths = []

for idx, f in enumerate(tqdm(eligible_files, desc="Budowanie datasetu")):
    file_id = file_id_from_name(f.name)
    y = eligible_labels[idx]

    img = nib.load(f)
    data = img.get_fdata().astype(np.float32)  # shape np. (61,73,61,T)

    # obcięcie czasu
    data = data[:, :, :, :TARGET_T]

    # resize przestrzenny dla każdego timepointu
    resized_time_series = np.empty((*TARGET_SHAPE, TARGET_T), dtype=np.float32)

    for t in range(TARGET_T):
        resized_time_series[:, :, :, t] = resize_3d_volume(data[:, :, :, t], TARGET_SHAPE)

    # standaryzacja per subject
    resized_time_series = standardize_subject(resized_time_series)

    X_list.append(resized_time_series)
    y_list.append(y)
    selected_paths.append(str(f))

    subject_rows.append({
        "idx": len(subject_rows),
        "FILE_ID": file_id,
        "path": str(f),
        "label_binary": y,
        "DX_GROUP_original": 1 if y == 1 else 2
    })

X = np.stack(X_list, axis=0).astype(np.float32)
y = np.array(y_list, dtype=np.int64)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Klasy [control=0, ASD=1]:", Counter(y.tolist()))

# =========================
# ZAPIS
# =========================
np.save(OUT_DIR / "X.npy", X)
np.save(OUT_DIR / "y.npy", y)

subject_index_df = pd.DataFrame(subject_rows)
subject_index_df.to_csv(OUT_DIR / "subject_index.csv", index=False)

with open(OUT_DIR / "selected_paths.txt", "w", encoding="utf-8") as f:
    for p in selected_paths:
        f.write(p + "\n")

info_lines = [
    f"TARGET_SHAPE={TARGET_SHAPE}",
    f"TARGET_T={TARGET_T}",
    f"N_SUBJECTS={len(y)}",
    f"N_CONTROL={(y == 0).sum()}",
    f"N_ASD={(y == 1).sum()}",
]

with open(OUT_DIR / "info.txt", "w", encoding="utf-8") as f:
    for line in info_lines:
        f.write(line + "\n")

print("\nZapisano:")
print(OUT_DIR / "X.npy")
print(OUT_DIR / "y.npy")
print(OUT_DIR / "subject_index.csv")
print(OUT_DIR / "selected_paths.txt")
print(OUT_DIR / "info.txt")