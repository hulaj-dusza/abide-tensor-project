from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# KONFIGURACJA
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\HP\Searches\fmri-tensor-project-git")

FALFF_DIR = (
    PROJECT_ROOT
    / "ABIDE_CPAC_3D_REHO_FALFF"
    / "Outputs"
    / "cpac"
    / "filt_noglobal"
    / "falff"
)

PHENOTYPIC_CSV = Path(
    r"C:\Users\HP\Searches\abide_data\ABIDE_pcp\Phenotypic_V1_0b_preprocessed1.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "61x73x61_falff_filt_noglobal_subjectwise"
X_DIR = OUTPUT_DIR / "X"

EXPECTED_SHAPE = (61, 73, 61)


# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================

def file_id_from_falff_path(path: Path) -> str:
    """
    Zamienia nazwę pliku:
        Caltech_0051456_falff.nii.gz
    na:
        Caltech_0051456
    """
    return path.name.replace("_falff.nii.gz", "")


def encode_dx_group(dx_group):
    """
    ABIDE:
        DX_GROUP = 1 -> ASD
        DX_GROUP = 2 -> CONTROL

    W modelu:
        y = 1 -> ASD
        y = 0 -> CONTROL
    """
    if dx_group == 1:
        return 1, "ASD"
    elif dx_group == 2:
        return 0, "CONTROL"
    else:
        return None, "UNKNOWN"


def load_falff_volume(path: Path) -> np.ndarray:
    """
    Wczytuje obraz fALFF 3D z pliku NIfTI i zwraca tablicę numpy float32.
    """
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)

    if data.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Niepoprawny shape dla {path.name}: {data.shape}, oczekiwano {EXPECTED_SHAPE}"
        )

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return data.astype(np.float32)


# ============================================================
# GŁÓWNY SKRYPT
# ============================================================

def main():
    print("=== Przygotowanie ABIDE fALFF 3D subjectwise ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"FALFF_DIR: {FALFF_DIR}")
    print(f"PHENOTYPIC_CSV: {PHENOTYPIC_CSV}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print()

    if not FALFF_DIR.exists():
        raise FileNotFoundError(f"Nie znaleziono folderu fALFF: {FALFF_DIR}")

    if not PHENOTYPIC_CSV.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku fenotypowego: {PHENOTYPIC_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Fenotypy
    # --------------------------------------------------------

    pheno = pd.read_csv(PHENOTYPIC_CSV)

    required_cols = ["FILE_ID", "DX_GROUP", "SITE_ID", "AGE_AT_SCAN", "SEX"]
    missing_cols = [c for c in required_cols if c not in pheno.columns]

    if missing_cols:
        raise ValueError(f"Brakuje kolumn w pliku fenotypowym: {missing_cols}")

    pheno = pheno.copy()
    pheno["FILE_ID"] = pheno["FILE_ID"].astype(str)

    # Usuwamy wiersze bez realnej nazwy pliku
    pheno = pheno[pheno["FILE_ID"] != "no_filename"].copy()

    pheno_by_file_id = pheno.set_index("FILE_ID", drop=False)

    print(f"Liczba wierszy fenotypowych po usunięciu no_filename: {len(pheno)}")

    # --------------------------------------------------------
    # Pliki fALFF
    # --------------------------------------------------------

    falff_files = sorted(FALFF_DIR.glob("*_falff.nii.gz"))

    print(f"Liczba plików fALFF znalezionych na dysku: {len(falff_files)}")

    if len(falff_files) == 0:
        raise RuntimeError(f"Nie znaleziono plików *_falff.nii.gz w {FALFF_DIR}")

    rows = []
    skipped = []

    for idx, falff_path in enumerate(falff_files, start=1):
        file_id = file_id_from_falff_path(falff_path)

        if file_id not in pheno_by_file_id.index:
            skipped.append(
                {
                    "file_id": file_id,
                    "nii_path": str(falff_path),
                    "reason": "missing_in_phenotypic_csv",
                }
            )
            continue

        row = pheno_by_file_id.loc[file_id]

        y, diagnosis = encode_dx_group(int(row["DX_GROUP"]))

        if y is None:
            skipped.append(
                {
                    "file_id": file_id,
                    "nii_path": str(falff_path),
                    "reason": f"unknown_dx_group_{row['DX_GROUP']}",
                }
            )
            continue

        try:
            X = load_falff_volume(falff_path)
        except Exception as e:
            skipped.append(
                {
                    "file_id": file_id,
                    "nii_path": str(falff_path),
                    "reason": f"load_error: {repr(e)}",
                }
            )
            continue

        out_name = f"{file_id}_falff.npy"
        out_path = X_DIR / out_name

        np.save(out_path, X)

        rows.append(
            {
                "file_id": file_id,
                "npy_path": str(out_path),
                "nii_path": str(falff_path),
                "y": y,
                "dx_group": int(row["DX_GROUP"]),
                "diagnosis": diagnosis,
                "site_id": row["SITE_ID"],
                "age_at_scan": row["AGE_AT_SCAN"],
                "sex": row["SEX"],
                "shape": str(X.shape),
                "derivative": "falff",
                "pipeline": "cpac",
                "strategy": "filt_noglobal",
            }
        )

        if idx % 50 == 0:
            print(f"Przetworzono {idx}/{len(falff_files)} plików...")

    # --------------------------------------------------------
    # Zapis indeksu
    # --------------------------------------------------------

    index_df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)

    index_path = OUTPUT_DIR / "subject_index_falff_filt_noglobal.csv"
    skipped_path = OUTPUT_DIR / "skipped_falff_filt_noglobal.csv"

    index_df.to_csv(index_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    print()
    print("=== Zakończono przygotowanie fALFF ===")
    print(f"Liczba zapisanych subjectów: {len(index_df)}")
    print(f"Liczba pominiętych plików: {len(skipped_df)}")
    print(f"Zapisano indeks: {index_path}")
    print(f"Zapisano skipped: {skipped_path}")

    if len(index_df) > 0:
        print()
        print("Rozkład klas:")
        print(index_df["diagnosis"].value_counts())

        print()
        print("Przykładowe wiersze:")
        print(index_df[["file_id", "y", "diagnosis", "site_id", "age_at_scan", "sex"]].head())


if __name__ == "__main__":
    main()