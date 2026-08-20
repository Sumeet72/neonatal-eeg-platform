from pathlib import Path
import pandas as pd
import numpy as np
import scipy.io as sio


# --------------------------------------------------
# PATHS
# --------------------------------------------------

NATHAN_ROOT = Path(
    r"C:\Users\Sumeet Mangat\Downloads\EEG_project\dataset\EEG\nathan  dataset\data"
)

ANNOTATION_DIR = NATHAN_ROOT / "labels"


# --------------------------------------------------
# FILES
# --------------------------------------------------

A_FILE = ANNOTATION_DIR / "annotations_2017_A_fixed.csv"
B_FILE = ANNOTATION_DIR / "annotations_2017_B.csv"
C_FILE = ANNOTATION_DIR / "annotations_2017_C.csv"

MAT_FILE = ANNOTATION_DIR / "annotations_2017.mat"

CLINICAL_FILE = ANNOTATION_DIR / "clinical_information.csv"


# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

files = [
    A_FILE,
    B_FILE,
    C_FILE,
    MAT_FILE,
    CLINICAL_FILE,
]

print("=" * 60)
print("NATHAN DATASET AUDIT")
print("=" * 60)

for file in files:
    print(f"{file.name:<35} {'FOUND' if file.exists() else 'MISSING'}")

if not all(file.exists() for file in files):
    raise FileNotFoundError(
        "\nOne or more Nathan dataset files could not be found."
    )


# --------------------------------------------------
# LOAD CSV FILES
# --------------------------------------------------

A = pd.read_csv(A_FILE)
B = pd.read_csv(B_FILE)
C = pd.read_csv(C_FILE)

clinical = pd.read_csv(CLINICAL_FILE)


# --------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------

print("\nDATASET SIZE")
print("-" * 60)

print(f"Subjects in clinical file : {len(clinical)}")
print(f"Annotation A shape        : {A.shape}")
print(f"Annotation B shape        : {B.shape}")
print(f"Annotation C shape        : {C.shape}")


# --------------------------------------------------
# SUBJECT IDs
# --------------------------------------------------

clinical_ids = clinical["ID"].astype(str).tolist()

annotation_ids = A.columns.astype(str).tolist()

print("\nID CHECK")
print("-" * 60)

print(f"Clinical IDs      : {len(clinical_ids)}")
print(f"Annotation IDs    : {len(annotation_ids)}")

print(
    "IDs match         :",
    set(clinical_ids) == set(annotation_ids)
)


# --------------------------------------------------
# ANNOTATION COUNTS
# --------------------------------------------------

print("\nANNOTATION COUNTS")
print("-" * 60)

for name, df in [
    ("A", A),
    ("B", B),
    ("C", C),
]:

    values = df.to_numpy(dtype=float)

    positive = np.nansum(values == 1)
    negative = np.nansum(values == 0)
    missing = np.isnan(values).sum()

    print(f"\nAnnotation {name}")
    print(f"Positive samples : {positive}")
    print(f"Negative samples : {negative}")
    print(f"Missing samples  : {missing}")


# --------------------------------------------------
# REVIEWER AGREEMENT
# --------------------------------------------------

print("\nREVIEWER AGREEMENT")
print("-" * 60)

agreement_counts = {
    "all_three_zero": 0,
    "all_three_one": 0,
    "two_positive": 0,
    "one_positive": 0,
    "disagreement": 0,
}

for subject in annotation_ids:

    a = A[subject].to_numpy(dtype=float)
    b = B[subject].to_numpy(dtype=float)
    c = C[subject].to_numpy(dtype=float)

    valid = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(c)

    a = a[valid]
    b = b[valid]
    c = c[valid]

    positive_reviewers = a + b + c

    agreement_counts["all_three_zero"] += np.sum(
        positive_reviewers == 0
    )

    agreement_counts["all_three_one"] += np.sum(
        positive_reviewers == 3
    )

    agreement_counts["two_positive"] += np.sum(
        positive_reviewers == 2
    )

    agreement_counts["one_positive"] += np.sum(
        positive_reviewers == 1
    )

    agreement_counts["disagreement"] += np.sum(
        (positive_reviewers > 0) &
        (positive_reviewers < 3)
    )


for key, value in agreement_counts.items():
    print(f"{key:<20}: {value}")


# --------------------------------------------------
# MAT FILE CHECK
# --------------------------------------------------

print("\nMAT FILE CHECK")
print("-" * 60)

mat = sio.loadmat(
    MAT_FILE,
    squeeze_me=True
)

annotations = mat["annotat_new"]

print(f"MAT subjects : {len(annotations)}")

for i in range(min(5, len(annotations))):

    subject_number = i + 1
    subject_annotations = annotations[i]

    print(
        f"Subject {subject_number:02d}: "
        f"shape={subject_annotations.shape}"
    )


# --------------------------------------------------
# EDF FILE CHECK
# --------------------------------------------------

print("\nEDF FILE CHECK")
print("-" * 60)

edf_files = sorted(
    NATHAN_ROOT.rglob("*.edf")
)

print(f"EDF files found : {len(edf_files)}")

print("\nFirst 10 EDF files:")

for file in edf_files[:10]:
    print(file.name)


# --------------------------------------------------
# CLINICAL INFORMATION
# --------------------------------------------------

print("\nCLINICAL INFORMATION")
print("-" * 60)

reviewer_column = (
    "Number of Reviewers Annotating Seizure"
)

if reviewer_column in clinical.columns:

    print(
        clinical[reviewer_column]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )


# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print("\n" + "=" * 60)
print("NATHAN AUDIT COMPLETE")
print("=" * 60)