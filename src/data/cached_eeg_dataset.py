import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedEEGDataset(Dataset):

    def __init__(self, metadata):

        if isinstance(
            metadata,
            str
        ):

            self.metadata = pd.read_csv(
                metadata
            )

        else:

            self.metadata = (
                metadata
                .reset_index(drop=True)
                .copy()
            )

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        spectrogram = torch.load(
            row["cache_file"],
            weights_only=True
        ).float()

        if tuple(
            spectrogram.shape
        ) != (9, 129, 59):

            raise RuntimeError(
                f"Invalid spectrogram shape: "
                f"{tuple(spectrogram.shape)}"
            )

        if not torch.isfinite(
            spectrogram
        ).all():

            raise RuntimeError(
                "NaN/Inf detected"
            )

        return {
            "spectrogram": spectrogram,
            "label": torch.tensor(
                int(row["label"]),
                dtype=torch.long
            ),
            "subject_id": str(
                row["subject_id"]
            ),
            "dataset": str(
                row["dataset"]
            ),
            "epoch_id": int(
                row["epoch_id"]
            )
        }


if __name__ == "__main__":

    dataset = CachedEEGDataset(
        "results/stft_cache/metadata.csv"
    )

    print(
        "Cached dataset size:",
        len(dataset)
    )

    sample = dataset[0]

    print(
        "Spectrogram:",
        sample["spectrogram"].shape
    )

    print(
        "Label:",
        sample["label"]
    )

    print(
        "Subject:",
        sample["subject_id"]
    )

    print(
        "\nCached EEG Dataset test successful."
    )