"""Cache corrected Sumit 30-second windows while reusing existing Nathan tensors.

Writes its metadata and new Sumit tensors under results/corrected_epoch_dataset/.
The original results/stft_cache is read-only in this script.
"""
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.eeg_dataset import EEGDataset, TARGET_RATE
from src.preprocessing.channel_selection import select_common_channels
from src.preprocessing.filtering import bandpass_filter
from src.preprocessing.resampling import resample_eeg
from src.preprocessing.artifact_detection import check_eeg_quality
from src.preprocessing.normalization import zscore_normalize
from src.transforms.stft import eeg_to_stft, log_power

ROOT = Path("results/corrected_epoch_dataset_car")
MASTER_FILE = Path("results/corrected_epoch_dataset/master_epochs.csv")
CACHE_DIR = ROOT / "stft_cache"
METADATA_FILE = CACHE_DIR / "metadata.csv"
EXPECTED_SHAPE = (9, 129, 59)
EXPECTED_SAMPLES = TARGET_RATE * 30


def save_metadata(df: pd.DataFrame) -> None:
    df = df.drop_duplicates(subset=["epoch_id"]).sort_values("epoch_id").reset_index(drop=True)
    df.to_csv(METADATA_FILE, index=False)


def make_spectrogram(raw, start_second: float, end_second: float) -> torch.Tensor:
    """Apply the same segment preprocessing/STFT operations as EEGDataset."""
    segment = raw.copy()
    try:
        segment.crop(tmin=start_second, tmax=end_second, include_tmax=False)
        segment.load_data()
        segment = select_common_channels(segment)
        original_rate = float(raw.info["sfreq"])
        eeg = segment.get_data()
        eeg = bandpass_filter(eeg, original_rate)
        eeg = resample_eeg(eeg, original_rate, TARGET_RATE)

        if eeg.shape[1] < EXPECTED_SAMPLES:
            padded = np.zeros((eeg.shape[0], EXPECTED_SAMPLES), dtype=np.float32)
            padded[:, :eeg.shape[1]] = eeg
            eeg = padded
        elif eeg.shape[1] > EXPECTED_SAMPLES:
            eeg = eeg[:, :EXPECTED_SAMPLES]

        # Match the existing loader's quality-check and normalization sequence.
        # Check signal quality before reference transformation.
        _quality = check_eeg_quality(eeg)

        # Experimental common-average reference across all 9 selected EEG channels.
        # Apply identically to Nathan and Sumit.
        eeg = eeg - np.mean(eeg, axis=0, keepdims=True)

        # Apply channel-wise Z-score normalization after CAR.
        eeg = zscore_normalize(eeg)
        eeg_tensor = torch.from_numpy(eeg).float()
        _, _, power = eeg_to_stft(eeg_tensor, sampling_rate=TARGET_RATE)
        spectrogram = log_power(power).cpu()
        if tuple(spectrogram.shape) != EXPECTED_SHAPE:
            raise RuntimeError(f"Invalid STFT shape: {tuple(spectrogram.shape)}")
        if not torch.isfinite(spectrogram).all():
            raise RuntimeError("STFT contains NaN or Inf")
        return spectrogram
    finally:
        try:
            segment.close()
        except Exception:
            pass


def main():
    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Corrected master manifest missing: {MASTER_FILE}"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(MASTER_FILE)

    required = {
        "epoch_id", "dataset", "subject_id",
        "edf_path", "start_second", "end_second", "label"
    }

    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    master["epoch_id"] = pd.to_numeric(
        master["epoch_id"], errors="raise"
    ).astype(int)

    master["dataset"] = master["dataset"].astype(str).str.lower()
    master["subject_id"] = master["subject_id"].astype(str)
    master["label"] = pd.to_numeric(
        master["label"], errors="raise"
    ).astype(int)

    if master["epoch_id"].duplicated().any():
        raise ValueError("Duplicate epoch IDs found.")

    if len(master) != 47015:
        raise ValueError(
            f"Expected 47,015 windows; found {len(master)}."
        )

    if set(master["dataset"].unique()) != {"nathan", "sumit"}:
        raise ValueError("Both Nathan and Sumit datasets must be present.")

    # Resume only from this new CAR cache.
    columns = [
        "cache_file", "epoch_id", "subject_id", "dataset", "label"
    ]

    if METADATA_FILE.exists():
        metadata = pd.read_csv(METADATA_FILE)

        if metadata["epoch_id"].duplicated().any():
            raise ValueError("Duplicate IDs in existing CAR metadata.")

        for path in metadata["cache_file"].astype(str):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Existing CAR tensor missing: {path}"
                )
    else:
        metadata = pd.DataFrame(columns=columns)

    completed = set(
        pd.to_numeric(metadata["epoch_id"], errors="raise")
        .astype(int).tolist()
    )

    expected_ids = set(master["epoch_id"].tolist())

    if not completed.issubset(expected_ids):
        raise RuntimeError("Existing CAR cache contains unexpected IDs.")

    remaining = master.loc[
        ~master["epoch_id"].isin(completed)
    ].copy()

    print("=" * 65)
    print("BUILDING NEW NATHAN + SUMIT CAR STFT CACHE")
    print("=" * 65)
    print("Total windows:", len(master))
    print("Existing CAR windows:", len(completed))
    print("Windows remaining:", len(remaining))
    print("Nathan windows:", int((master["dataset"] == "nathan").sum()))
    print("Sumit windows:", int((master["dataset"] == "sumit").sum()))
    print("Output:", CACHE_DIR)

    failures = []

    groups = list(
        remaining.groupby("edf_path", sort=False)
    )

    for edf_path_str, rows in tqdm(
        groups, desc="Building CAR spectrograms"
    ):
        raw = None
        recording_entries = []

        try:
            edf_path = Path(str(edf_path_str))

            if not edf_path.exists():
                raise FileNotFoundError(edf_path)

            raw = mne.io.read_raw_edf(
                str(edf_path),
                preload=False,
                verbose="ERROR",
            )

            duration = (
                raw.n_times / float(raw.info["sfreq"])
            )

            for row in rows.itertuples(index=False):
                try:
                    if float(row.end_second) > duration + 1e-3:
                        raise ValueError(
                            f"Window exceeds EDF duration: {row.end_second}s"
                        )

                    spectrogram = make_spectrogram(
                        raw,
                        float(row.start_second),
                        float(row.end_second),
                    )

                    cache_file = CACHE_DIR / f"{int(row.epoch_id):05d}.pt"

                    torch.save(spectrogram, cache_file)

                    recording_entries.append({
                        "cache_file": str(cache_file),
                        "epoch_id": int(row.epoch_id),
                        "subject_id": str(row.subject_id),
                        "dataset": str(row.dataset),
                        "label": int(row.label),
                    })

                except Exception as exc:
                    failures.append({
                        "epoch_id": int(row.epoch_id),
                        "edf_path": str(edf_path_str),
                        "error": repr(exc),
                    })

        except Exception as exc:
            for row in rows.itertuples(index=False):
                failures.append({
                    "epoch_id": int(row.epoch_id),
                    "edf_path": str(edf_path_str),
                    "error": repr(exc),
                })

        finally:
            if raw is not None:
                raw.close()

        if recording_entries:
            metadata = pd.concat(
                [metadata, pd.DataFrame(recording_entries)],
                ignore_index=True,
            )

            metadata = metadata.drop_duplicates(
                subset=["epoch_id"]
            )

            metadata = metadata.sort_values(
                "epoch_id"
            ).reset_index(drop=True)

            metadata.to_csv(METADATA_FILE, index=False)

    if failures:
        failures_file = ROOT / "car_cache_failures.csv"
        pd.DataFrame(failures).to_csv(
            failures_file, index=False
        )
        print("Failed windows:", len(failures))
        print("Failure report:", failures_file)

    if METADATA_FILE.exists():
        final_meta = pd.read_csv(METADATA_FILE)
        final_ids = set(
            pd.to_numeric(final_meta["epoch_id"], errors="raise")
            .astype(int).tolist()
        )
    else:
        final_ids = set()

    missing_ids = expected_ids - final_ids

    print("\n" + "=" * 65)
    print("CAR CACHE BUILD SUMMARY")
    print("=" * 65)
    print("Cached windows:", len(final_ids), "/", len(expected_ids))
    print("Missing windows:", len(missing_ids))

    if METADATA_FILE.exists():
        print(
            "Nathan cached:",
            int((final_meta["dataset"] == "nathan").sum())
        )
        print(
            "Sumit cached:",
            int((final_meta["dataset"] == "sumit").sum())
        )

    print("CAR cache metadata:", METADATA_FILE)
    print("Original caches and checkpoints were not modified.")

    if missing_ids:
        print("CACHE INCOMPLETE. Review failures before training.")
    else:
        print("CAR CACHE COMPLETE.")


if __name__ == "__main__":
    main()