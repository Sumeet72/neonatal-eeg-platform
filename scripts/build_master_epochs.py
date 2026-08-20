from pathlib import Path
import pandas as pd


nathan_file = Path("results/nathan_epochs.csv")
sumit_file = Path("results/sumit_epochs.csv")


if not nathan_file.exists():
    raise FileNotFoundError(
        f"Missing: {nathan_file}"
    )

if not sumit_file.exists():
    raise FileNotFoundError(
        f"Missing: {sumit_file}"
    )


nathan = pd.read_csv(nathan_file)
sumit = pd.read_csv(sumit_file)


print("Nathan epochs:", len(nathan))
print("Sumit epochs:", len(sumit))


# Keep only columns that are common to both datasets.
#
# We deliberately do not force Nathan-specific columns
# such as subject_number into the Sumit dataset.
common_columns = [
    "dataset",
    "edf_path",
    "label"
]


# Create a common subject identifier.
#
# Nathan already has subject_id.
# Sumit uses baby_ID.

nathan_master = pd.DataFrame({
    "dataset": "nathan",
    "subject_id": nathan["subject_id"],
    "edf_path": nathan["edf_path"],
    "start_second": nathan["start_second"],
    "end_second": nathan["end_second"],
    "label": nathan["label"]
})


sumit_master = pd.DataFrame({
    "dataset": "sumit",
    "subject_id": sumit["baby_ID"],
    "edf_path": sumit["edf_path"],
    "start_second": (
        (sumit["epoch_number"] - 1) * 30
    ),
    "end_second": (
        sumit["epoch_number"] * 30
    ),
    "label": sumit["label"]
})


master = pd.concat(
    [
        nathan_master,
        sumit_master
    ],
    ignore_index=True
)


# Add a unique epoch ID
master.insert(
    0,
    "epoch_id",
    range(len(master))
)


# Make sure labels are integers
master["label"] = (
    master["label"]
    .astype(int)
)


# Validate
if master["edf_path"].isna().any():
    raise ValueError(
        "Some epochs have missing EDF paths."
    )

if master["label"].isna().any():
    raise ValueError(
        "Some epochs have missing labels."
    )


output_dir = Path("results")
output_dir.mkdir(exist_ok=True)


output_file = output_dir / "master_epochs.csv"


master.to_csv(
    output_file,
    index=False
)


print("\n=== MASTER DATASET ===")

print(
    "Total epochs:",
    len(master)
)

print(
    "Nathan epochs:",
    (master["dataset"] == "nathan").sum()
)

print(
    "Sumit epochs:",
    (master["dataset"] == "sumit").sum()
)

print(
    "Unique subjects:",
    master["subject_id"].nunique()
)

print("\nLabels:")
print(master["label"].value_counts())

print("\nDataset × Label:")
print(
    pd.crosstab(
        master["dataset"],
        master["label"]
    )
)

print("\nFirst 5 rows:")
print(
    master.head().to_string(
        index=False
    )
)

print("\nSaved to:")
print(output_file)