from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.data.eeg_dataset import EEGDataset


MASTER_FILE = "results/master_epochs.csv"

CACHE_DIR = Path("results/stft_cache")

METADATA_FILE = CACHE_DIR / "metadata.csv"

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    MASTER_FILE
)

print("Master dataset:", df.shape)
print("Total epochs:", len(df))


if METADATA_FILE.exists():

    existing = pd.read_csv(
        METADATA_FILE
    )

    completed_ids = set(
        existing["epoch_id"].astype(int)
    )

    print(
        "Already cached:",
        len(completed_ids)
    )

else:

    existing = pd.DataFrame(
        columns=[
            "cache_file",
            "epoch_id",
            "subject_id",
            "dataset",
            "label"
        ]
    )

    completed_ids = set()

    print("No existing cache found.")


remaining_df = df[
    ~df["epoch_id"].isin(completed_ids)
].copy()

remaining_df = remaining_df.reset_index(
    drop=True
)

print(
    "Remaining epochs:",
    len(remaining_df)
)


if len(remaining_df) == 0:

    print(
        "\nAll epochs are already cached."
    )

    raise SystemExit


dataset = EEGDataset(
    remaining_df,
    device="cpu"
)


new_metadata = []
failed = []


for i in tqdm(
    range(len(dataset)),
    desc="Caching STFT"
):

    row = remaining_df.iloc[i]

    epoch_id = int(
        row["epoch_id"]
    )

    try:

        sample = dataset[i]

        spectrogram = sample[
            "spectrogram"
        ]

        label = int(
            sample["label"].item()
        )

        subject_id = sample[
            "subject_id"
        ]

        dataset_name = sample[
            "dataset"
        ]

        expected_shape = (
            9,
            129,
            59
        )

        if tuple(
            spectrogram.shape
        ) != expected_shape:

            raise RuntimeError(
                f"Invalid shape: "
                f"{tuple(spectrogram.shape)}"
            )

        if not torch.isfinite(
            spectrogram
        ).all():

            raise RuntimeError(
                "NaN/Inf detected"
            )

        cache_file = (
            CACHE_DIR
            / f"{epoch_id:05d}.pt"
        )

        torch.save(
            spectrogram.cpu(),
            cache_file
        )

        new_metadata.append(
            {
                "cache_file": str(
                    cache_file
                ),
                "epoch_id": epoch_id,
                "subject_id": subject_id,
                "dataset": dataset_name,
                "label": label
            }
        )

    except Exception as e:

        failed.append(
            {
                "epoch_id": epoch_id,
                "error": str(e)
            }
        )

        print(
            f"\nFAILED epoch "
            f"{epoch_id}: {e}"
        )


if new_metadata:

    new_df = pd.DataFrame(
        new_metadata
    )

    combined = pd.concat(
        [
            existing,
            new_df
        ],
        ignore_index=True
    )

    combined = (
        combined
        .drop_duplicates(
            subset=["epoch_id"]
        )
        .sort_values(
            "epoch_id"
        )
        .reset_index(drop=True)
    )

    combined.to_csv(
        METADATA_FILE,
        index=False
    )

else:

    combined = existing


if failed:

    failed_df = pd.DataFrame(
        failed
    )

    failed_file = (
        CACHE_DIR
        / "failed_epochs.csv"
    )

    failed_df.to_csv(
        failed_file,
        index=False
    )

    print(
        "\nFailed epochs:",
        len(failed)
    )

    print(
        "Failure report:",
        failed_file
    )


print(
    "\n============================================================"
)

print(
    "STFT CACHE RUN COMPLETE"
)

print(
    "Successfully cached:",
    len(new_metadata)
)

print(
    "Total cached:",
    len(combined)
)

print(
    "Failed:",
    len(failed)
)

print(
    "Metadata:",
    METADATA_FILE
)

print(
    "Cache directory:",
    CACHE_DIR
)
