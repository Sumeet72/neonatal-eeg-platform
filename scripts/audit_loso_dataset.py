import pandas as pd
import numpy as np


MASTER_FILE = "results/master_epochs.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    MASTER_FILE
)

print("============================================================")
print("LOSO DATASET AUDIT")
print("============================================================")

print(
    "Total epochs:",
    len(df)
)

print(
    "Total subjects:",
    df["subject_id"].nunique()
)


# ============================================================
# SUBJECT SUMMARY
# ============================================================

subject_summary = (
    df.groupby("subject_id")
    .agg(
        total_epochs=("label", "size"),
        seizure_epochs=("label", "sum"),
        normal_epochs=("label", lambda x: (x == 0).sum())
    )
    .reset_index()
)


subject_summary["seizure_prevalence"] = (
    subject_summary["seizure_epochs"]
    / subject_summary["total_epochs"]
)


# ============================================================
# SUBJECT TYPES
# ============================================================

zero_seizure = subject_summary[
    subject_summary["seizure_epochs"] == 0
]

with_seizure = subject_summary[
    subject_summary["seizure_epochs"] > 0
]


print("\n============================================================")
print("SUBJECT COMPOSITION")
print("============================================================")

print(
    "Subjects with seizure epochs:",
    len(with_seizure)
)

print(
    "Subjects without seizure epochs:",
    len(zero_seizure)
)


# ============================================================
# EPOCH DISTRIBUTION
# ============================================================

print("\n============================================================")
print("EPOCH DISTRIBUTION")
print("============================================================")

print(
    "Total normal epochs:",
    int(df["label"].eq(0).sum())
)

print(
    "Total seizure epochs:",
    int(df["label"].eq(1).sum())
)


# ============================================================
# SEIZURE SUBJECT DISTRIBUTION
# ============================================================

print("\n============================================================")
print("SEIZURE EPOCHS PER POSITIVE SUBJECT")
print("============================================================")

print(
    with_seizure[
        "seizure_epochs"
    ].describe().to_string()
)


# ============================================================
# SOURCE DATASET
# ============================================================

if "dataset" in df.columns:

    print("\n============================================================")
    print("DATASET SOURCES")
    print("============================================================")

    print(
        df.groupby("dataset")
        .agg(
            subjects=("subject_id", "nunique"),
            epochs=("label", "size"),
            seizures=("label", "sum")
        )
        .to_string()
    )


# ============================================================
# SUBJECT PREFIX
# ============================================================

subject_strings = (
    subject_summary["subject_id"]
    .astype(str)
)

id_subjects = subject_summary[
    subject_strings.str.startswith("ID")
]

n_subjects = subject_summary[
    subject_strings.str.startswith("N")
]


print("\n============================================================")
print("SUBJECT ID GROUPS")
print("============================================================")

print(
    "IDxx subjects:",
    len(id_subjects)
)

print(
    "Nxxx subjects:",
    len(n_subjects)
)


# ============================================================
# ID GROUP DETAILS
# ============================================================

if len(id_subjects) > 0:

    print("\nIDxx group:")

    print(
        id_subjects[
            [
                "subject_id",
                "total_epochs",
                "seizure_epochs",
                "normal_epochs"
            ]
        ]
        .sort_values("subject_id")
        .to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

output_file = (
    "results/loso_dataset_audit.csv"
)

subject_summary.to_csv(
    output_file,
    index=False
)


print(
    "\nAudit saved:",
    output_file
)

print(
    "\n============================================================"
)

print(
    "DATASET AUDIT COMPLETE"
)

print(
    "============================================================"
)