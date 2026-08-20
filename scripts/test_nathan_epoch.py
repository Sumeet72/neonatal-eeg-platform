from pathlib import Path
import yaml
import pandas as pd

from src.data.nathan_dataset import NathanDatasetInfo
from src.preprocessing.edf_epoch_extractor import extract_epochs


with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)


# Load Nathan annotations and EDF mapping
dataset = NathanDatasetInfo(
    edf_dir=config["nathan"]["edf_dir"],
    annotations_dir=config["nathan"]["annotations_dir"]
)

dataset.load_annotations()
dataset.validate_annotations()
dataset.create_majority_labels()
dataset.find_edf_files()


# Test subject 1
subject_id = 1

edf_path = dataset.edf_map[subject_id]
second_labels = dataset.subject_labels[subject_id]


print("EDF:", edf_path)
print("Annotation seconds:", len(second_labels))


epochs = extract_epochs(
    edf_path=edf_path,
    second_labels=second_labels,
    epoch_seconds=30,
    stride_seconds=15,
    seizure_threshold=0.5
)


print("Number of epochs:", len(epochs))

if epochs:

    first = epochs[0]

    print("First epoch:")
    print("Start:", first["start_second"])
    print("End:", first["end_second"])
    print("Label:", first["label"])
    print("EEG shape:", first["eeg"].shape)