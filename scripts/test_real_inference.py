import pandas as pd
import torch

from src.data.cached_eeg_dataset import CachedEEGDataset
from src.inference import EEGInference


# ============================================================
# LOAD CACHED DATA
# ============================================================

METADATA_FILE = "results/stft_cache/metadata.csv"

df = pd.read_csv(
    METADATA_FILE
)

print("=" * 70)
print("REAL EEG INFERENCE TEST")
print("=" * 70)

print(
    "Total cached epochs:",
    len(df)
)


# ============================================================
# SELECT ONE NORMAL + ONE SEIZURE SAMPLE
# ============================================================

normal_df = df[
    df["label"] == 0
].iloc[:1].copy()

seizure_df = df[
    df["label"] == 1
].iloc[:1].copy()


# ============================================================
# CREATE DATASETS
# ============================================================

normal_dataset = CachedEEGDataset(
    normal_df
)

seizure_dataset = CachedEEGDataset(
    seizure_df
)


# ============================================================
# LOAD INFERENCE MODEL
# ============================================================

inference = EEGInference()


# ============================================================
# TEST NORMAL SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("NORMAL SAMPLE")
print("=" * 70)

normal_sample = normal_dataset[0]

normal_spectrogram = (
    normal_sample["spectrogram"]
)

print(
    "Subject:",
    normal_sample["subject_id"]
)

print(
    "Epoch:",
    normal_sample["epoch_id"]
)

print(
    "True label:",
    normal_sample["label"].item()
)

print(
    "Input shape:",
    tuple(normal_spectrogram.shape)
)

normal_result = inference.predict(
    normal_spectrogram
)

print(
    "Prediction:",
    normal_result
)


# ============================================================
# TEST SEIZURE SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("SEIZURE SAMPLE")
print("=" * 70)

seizure_sample = seizure_dataset[0]

seizure_spectrogram = (
    seizure_sample["spectrogram"]
)

print(
    "Subject:",
    seizure_sample["subject_id"]
)

print(
    "Epoch:",
    seizure_sample["epoch_id"]
)

print(
    "True label:",
    seizure_sample["label"].item()
)

print(
    "Input shape:",
    tuple(seizure_spectrogram.shape)
)

seizure_result = inference.predict(
    seizure_spectrogram
)

print(
    "Prediction:",
    seizure_result
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("REAL INFERENCE TEST COMPLETE")
print("=" * 70)