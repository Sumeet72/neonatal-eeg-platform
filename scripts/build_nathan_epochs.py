from pathlib import Path
import yaml
import pandas as pd

from src.data.nathan_dataset import NathanDatasetInfo
from src.preprocessing.edf_epoch_extractor import extract_epochs


with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)


dataset = NathanDatasetInfo(
    edf_dir=config["nathan"]["edf_dir"],
    annotations_dir=config["nathan"]["annotations_dir"]
)

dataset.load_annotations()
dataset.validate_annotations()
dataset.create_majority_labels()
dataset.find_edf_files()


rows = []


for subject_id, edf_path in sorted(dataset.edf_map.items()):

    print(
        f"Processing subject {subject_id}: "
        f"{Path(edf_path).name}"
    )

    second_labels = dataset.subject_labels[subject_id]

    epochs = extract_epochs(
        edf_path=edf_path,
        second_labels=second_labels,
        epoch_seconds=30,
        stride_seconds=15,
        seizure_threshold=0.5
    )

    for epoch in epochs:

        rows.append(
            {
                "dataset": "nathan",
                "subject_id": f"N{subject_id:03d}",
                "subject_number": subject_id,
                "edf_path": edf_path,
                "start_second": epoch["start_second"],
                "end_second": epoch["end_second"],
                "label": epoch["label"]
            }
        )


df = pd.DataFrame(rows)


output_dir = Path("results")
output_dir.mkdir(exist_ok=True)


output_file = output_dir / "nathan_epochs.csv"

df.to_csv(
    output_file,
    index=False
)


print("\n=== NATHAN EPOCH SUMMARY ===")

print("Total epochs:", len(df))

print(
    "Unique subjects:",
    df["subject_id"].nunique()
)

print("\nLabels:")

print(
    df["label"].value_counts()
)

print("\nSaved to:")

print(output_file)