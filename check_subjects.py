from pathlib import Path
from collections import Counter, defaultdict
import nibabel as nib
import pandas as pd

# folder z pobranymi plikami fMRI
folder = Path(r"C:\Users\HP\fmri_tensor_project\abide_data\ABIDE_pcp\cpac\filt_global")
files = list(folder.glob("*.nii.gz"))

print("Liczba subjectów:", len(files))

# ===== znajdź lokalny plik fenotypowy =====
candidate_csvs = [
    Path(r"C:\Users\HP\fmri_tensor_project\abide_data\ABIDE_pcp\Phenotypic_V1_0b_preprocessed1.csv"),
    Path("Phenotypic_V1_0b_preprocessed1.csv"),
    Path("abide_data/Phenotypic_V1_0b_preprocessed1.csv"),
    Path("abide_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"),
]

pheno_path = None
for p in candidate_csvs:
    if p.exists():
        pheno_path = p
        break

if pheno_path is None:
    print("\nNie znalazłam lokalnego pliku fenotypowego CSV.")
    print("Sprawdzę tylko shape i timepoints.")
    df = None
else:
    print(f"\nUżywam lokalnego CSV: {pheno_path}")
    df = pd.read_csv(pheno_path)

    # ujednolicenie FILE_ID do stringa
    if "FILE_ID" in df.columns:
        df["FILE_ID"] = df["FILE_ID"].astype(str)

# ===== analiza plików =====
shapes = []
timepoints = []
global_dx = []
time_class_counter = defaultdict(list)

for f in files:
    img = nib.load(f)
    shape = img.header.get_data_shape()

    shapes.append(shape)

    if len(shape) >= 4:
        t = shape[3]
        timepoints.append(t)
    else:
        t = None

    if df is not None and "FILE_ID" in df.columns and "DX_GROUP" in df.columns:
        file_id = f.name.split("_func")[0]
        row = df[df["FILE_ID"] == file_id]

        if not row.empty:
            dx = int(row["DX_GROUP"].iloc[0])
            global_dx.append(dx)
            if t is not None:
                time_class_counter[t].append(dx)

print("\nRóżne shape:")
print(Counter(shapes))

print("\nTimepoints:")
print(Counter(timepoints))

if df is not None and global_dx:
    dx_counter = Counter(global_dx)
    print("\nDX_GROUP (globalnie):")
    print(f"ASD (1): {dx_counter.get(1, 0)}")
    print(f"Control (2): {dx_counter.get(2, 0)}")

    print("\nDX_GROUP per timepoint:")
    for t in sorted(time_class_counter.keys()):
        counter = Counter(time_class_counter[t])
        print(f"\nTime = {t}")
        print(f"ASD (1): {counter.get(1, 0)}")
        print(f"Control (2): {counter.get(2, 0)}")
else:
    print("\nBrak lokalnego mapowania klas FILE_ID -> DX_GROUP, więc wypisano tylko shape i timepoints.")