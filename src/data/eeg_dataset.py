from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.channel_selection import select_common_channels
from src.preprocessing.filtering import bandpass_filter
from src.preprocessing.resampling import resample_eeg
from src.preprocessing.artifact_detection import check_eeg_quality
from src.preprocessing.normalization import zscore_normalize
from src.transforms.stft import eeg_to_stft, log_power


TARGET_RATE = 256

CHANNELS = [
    "F3",
    "F4",
    "C3",
    "C4",
    "O1",
    "O2",
    "T3",
    "T4",
    "Cz"
]


class EEGDataset(Dataset):

    def __init__(
        self,
        dataframe,
        device="cpu"
    ):
        self.dataframe = (
            dataframe.reset_index(drop=True)
        )

        self.device = torch.device(device)

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        edf_path = Path(
            row["edf_path"]
        )

        start_second = float(
            row["start_second"]
        )

        end_second = float(
            row["end_second"]
        )

        label = int(
            row["label"]
        )

        subject_id = str(
            row["subject_id"]
        )

        # ----------------------------------------
        # 1. Read only required EDF segment
        # ----------------------------------------

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=False,
            verbose=False
        )

        original_rate = float(
            raw.info["sfreq"]
        )

        start_sample = int(
            start_second * original_rate
        )

        stop_sample = int(
            end_second * original_rate
        )

        raw.crop(
            tmin=start_second,
            tmax=end_second,
            include_tmax=False
        )

        raw.load_data()

        # ----------------------------------------
        # 2. Select common EEG channels
        # ----------------------------------------

        raw = select_common_channels(
            raw
        )

        eeg = raw.get_data()

        # ----------------------------------------
        # 3. Filter
        # ----------------------------------------

        eeg = bandpass_filter(
            eeg,
            original_rate
        )

        # ----------------------------------------
        # 4. Resample
        # ----------------------------------------

        eeg = resample_eeg(
            eeg,
            original_rate,
            TARGET_RATE
        )

        # ----------------------------------------
        # 5. Ensure exactly 30 sec / 7680 samples
        # ----------------------------------------

        expected_samples = (
            TARGET_RATE * 30
        )

        if eeg.shape[1] < expected_samples:

            padded = np.zeros(
                (
                    eeg.shape[0],
                    expected_samples
                ),
                dtype=np.float32
            )

            padded[:, :eeg.shape[1]] = eeg

            eeg = padded

        elif eeg.shape[1] > expected_samples:

            eeg = eeg[
                :, :expected_samples
            ]

        # ----------------------------------------
        # 6. Quality check
        # ----------------------------------------

        quality = check_eeg_quality(
            eeg
        )

        # ----------------------------------------
        # 7. Normalize
        # ----------------------------------------

        eeg = zscore_normalize(
            eeg
        )

        # ----------------------------------------
        # 8. Convert to PyTorch
        # ----------------------------------------

        eeg_tensor = torch.from_numpy(
            eeg
        ).float()

        # ----------------------------------------
        # 9. STFT
        # ----------------------------------------

        frequencies, times, power = eeg_to_stft(
            eeg_tensor,
            sampling_rate=TARGET_RATE
        )

        spectrogram = log_power(
            power
        )

        # ----------------------------------------
        # 10. Return metadata + tensor
        # ----------------------------------------

        return {
    "spectrogram": spectrogram,
    "label": torch.tensor(
        label,
        dtype=torch.long
    ),
    "subject_id": subject_id,
    "dataset": str(
        row["dataset"]
    ),
    "epoch_id": int(
        row["epoch_id"]
    )
}


if __name__ == "__main__":

    master_path = (
        "results/master_epochs.csv"
    )

    df = pd.read_csv(
        master_path
    )

    print(
        "Master dataset:",
        df.shape
    )

    dataset = EEGDataset(
        df.iloc[:1]
    )

    sample = dataset[0]

    print(
        "\nSample loaded successfully."
    )

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
        "Dataset:",
        sample["dataset"]
    )

    print(
        "Epoch:",
        sample["epoch_id"]
    )