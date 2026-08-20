from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# PATHS
# --------------------------------------------------

SUMIT_ROOT = Path(
    r"C:\Users\Sumeet Mangat\Downloads\EEG_project\dataset\EEG\sumit dataset"
)

DATA_DIR = SUMIT_ROOT / "data"
LABEL_DIR = SUMIT_ROOT / "labels"

GRADE_FILE = LABEL_DIR / "eeggrades" / "eeg_grades.csv"
METADATA_FILE = LABEL_DIR / "metadata" / "metadata.csv"


# --------------------------------------------------
# HEADER
# --------------------------------------------------

print("=" * 60)
print("SUMIT DATASET AUDIT")
print("=" * 60)


# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

files = [
    DATA_DIR,
    GRADE_FILE,
    METADATA_FILE,
]

for file in files:
    print(
        f"{file.name:<35} "
        f"{'FOUND' if file.exists() else 'MISSING'}"
    )

if not all(file.exists() for file in files):
    raise FileNotFoundError(
        "\nOne or more Sumit dataset paths could not be found."
    )


# --------------------------------------------------
# FIND EDF FILES
# --------------------------------------------------

edf_files = sorted(
    DATA_DIR.rglob("*.edf")
)

print("\nEDF INFORMATION")
print("-" * 60)

print(f"EDF files found : {len(edf_files)}")

print("\nFirst 20 EDF files:")

for file in edf_files[:20]:
    print(file.name)


# --------------------------------------------------
# LOAD LABEL FILES
# --------------------------------------------------

grades = pd.read_csv(GRADE_FILE)
metadata = pd.read_csv(METADATA_FILE)


# --------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------

print("\nLABEL FILES")
print("-" * 60)

print(f"eeg_grades rows : {len(grades)}")
print(f"metadata rows   : {len(metadata)}")

print(f"\neeg_grades columns:")
print(grades.columns.tolist())

print(f"\nmetadata columns:")
print(metadata.columns.tolist())


# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_grade_columns = {
    "file_ID",
    "baby_ID",
    "epoch_number",
    "grade",
}

required_metadata_columns = {
    "file_ID",
    "baby_ID",
    "epoch_number",
    "grade",
    "sampling_freq",
    "reference",
    "EEG_quality_comment",
    "seizures_YN",
    "seizures_comment",
}

print("\nCOLUMN CHECK")
print("-" * 60)

print(
    "eeg_grades columns OK :",
    required_grade_columns.issubset(grades.columns)
)

print(
    "metadata columns OK   :",
    required_metadata_columns.issubset(metadata.columns)
)


# --------------------------------------------------
# SUBJECT COUNT
# --------------------------------------------------

print("\nSUBJECT INFORMATION")
print("-" * 60)

grade_subjects = grades["baby_ID"].nunique()
metadata_subjects = metadata["baby_ID"].nunique()

print(f"Unique babies in grades   : {grade_subjects}")
print(f"Unique babies in metadata : {metadata_subjects}")


# --------------------------------------------------
# EPOCH COUNT
# --------------------------------------------------

print("\nEPOCH INFORMATION")
print("-" * 60)

print(f"Total grade epochs    : {len(grades)}")
print(f"Total metadata epochs : {len(metadata)}")

print(
    "file_ID sets match    :",
    set(grades["file_ID"]) == set(metadata["file_ID"])
)


# --------------------------------------------------
# GRADE DISTRIBUTION
# --------------------------------------------------

print("\nHIE GRADE DISTRIBUTION")
print("-" * 60)

print(
    metadata["grade"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)


# --------------------------------------------------
# SEIZURE DISTRIBUTION
# --------------------------------------------------

print("\nSEIZURE DISTRIBUTION")
print("-" * 60)

print(
    metadata["seizures_YN"]
    .value_counts(dropna=False)
    .to_string()
)


# --------------------------------------------------
# SAMPLING FREQUENCY
# --------------------------------------------------

print("\nSAMPLING FREQUENCY")
print("-" * 60)

print(
    metadata["sampling_freq"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)


# --------------------------------------------------
# REFERENCE
# --------------------------------------------------

print("\nREFERENCE DISTRIBUTION")
print("-" * 60)

print(
    metadata["reference"]
    .value_counts(dropna=False)
    .to_string()
)


# --------------------------------------------------
# EEG QUALITY
# --------------------------------------------------

print("\nEEG QUALITY COMMENTS")
print("-" * 60)

print(
    metadata["EEG_quality_comment"]
    .value_counts(dropna=False)
    .to_string()
)


# --------------------------------------------------
# CHECK EDF ↔ CSV MAPPING
# --------------------------------------------------

print("\nEDF ↔ LABEL MAPPING")
print("-" * 60)

edf_names = {
    file.stem
    for file in edf_files
}

csv_ids = set(
    metadata["file_ID"]
    .astype(str)
)

matched = csv_ids.intersection(edf_names)
missing_edf = csv_ids - edf_names
extra_edf = edf_names - csv_ids

print(f"CSV file_IDs       : {len(csv_ids)}")
print(f"Matching EDFs      : {len(matched)}")
print(f"CSV IDs without EDF: {len(missing_edf)}")
print(f"EDFs without CSV   : {len(extra_edf)}")

if missing_edf:
    print("\nCSV IDs without EDF:")
    for item in sorted(missing_edf):
        print(item)

if extra_edf:
    print("\nEDFs without CSV:")
    for item in sorted(extra_edf):
        print(item)


# --------------------------------------------------
# DUPLICATE CHECK
# --------------------------------------------------

print("\nDUPLICATE CHECK")
print("-" * 60)

duplicates = metadata.duplicated(
    subset=["file_ID", "baby_ID", "epoch_number"]
)

print(
    "Duplicate epoch records:",
    duplicates.sum()
)


# --------------------------------------------------
# FINAL
# --------------------------------------------------

print("\n" + "=" * 60)
print("SUMIT AUDIT COMPLETE")
print("=" * 60)