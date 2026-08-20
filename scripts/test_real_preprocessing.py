from pathlib import Path
import yaml
import mne
import numpy as np
import torch

from src.preprocessing.channel_selection import select_common_channels
from src.preprocessing.filtering import bandpass_filter
from src.preprocessing.resampling import resample_eeg
from src.preprocessing.artifact_detection import check_eeg_quality
from src.preprocessing.normalization import zscore_normalize
from src.transforms.stft import eeg_to_stft, log_power
from src.utils.device import get_device


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

TARGET_RATE = 256
EPOCH_DURATION = 30

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


# --------------------------------------------------
# LOAD PATHS
# --------------------------------------------------

with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)


# --------------------------------------------------
# GET ONE EDF
# --------------------------------------------------

nathan_dir = Path(
    config["nathan"]["edf_dir"]
)

sumit_dir = Path(
    config["sumit"]["edf_dir"]
)

nathan_file = sorted(
    nathan_dir.rglob("*.edf")
)[0]

sumit_file = sorted(
    sumit_dir.rglob("*.edf")
)[0]


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = get_device()

print("\nSelected device:", device)


# --------------------------------------------------
# PROCESS ONE REAL EDF
# --------------------------------------------------

def process_real_edf(
    edf_path,
    dataset_name
):

    print("\n" + "=" * 60)
    print(dataset_name.upper())
    print("=" * 60)

    print("EDF:", edf_path)

    # ----------------------------------------------
    # 1. LOAD EDF
    # ----------------------------------------------

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False
    )

    print(
        "Original sampling rate:",
        raw.info["sfreq"]
    )

    print(
        "Original channels:",
        len(raw.ch_names)
    )

    # ----------------------------------------------
    # 2. SELECT COMMON CHANNELS
    # ----------------------------------------------

    raw = select_common_channels(
        raw
    )

    print(
        "Selected channels:",
        raw.ch_names
    )

    # ----------------------------------------------
    # 3. GET EEG DATA
    # ----------------------------------------------

    eeg = raw.get_data()

    print(
        "Original EEG shape:",
        eeg.shape
    )

    original_rate = raw.info["sfreq"]

    # ----------------------------------------------
    # 4. FILTER
    # ----------------------------------------------

    eeg = bandpass_filter(
        eeg,
        original_rate
    )

    print(
        "After filtering:",
        eeg.shape
    )

    # ----------------------------------------------
    # 5. RESAMPLE
    # ----------------------------------------------

    eeg = resample_eeg(
        eeg,
        original_rate,
        TARGET_RATE
    )

    print(
        "After resampling:",
        eeg.shape
    )

    # ----------------------------------------------
    # 6. EXTRACT FIRST 30-SECOND EPOCH
    # ----------------------------------------------

    samples_per_epoch = (
        TARGET_RATE * EPOCH_DURATION
    )

    if eeg.shape[1] < samples_per_epoch:

        print(
            "ERROR: EDF is shorter than 30 seconds."
        )

        return

    epoch = eeg[
        :,
        :samples_per_epoch
    ]

    print(
        "30-second epoch shape:",
        epoch.shape
    )

    # ----------------------------------------------
    # 7. QUALITY CHECK
    # ----------------------------------------------

    quality = check_eeg_quality(
        epoch
    )

    print(
        "Quality:",
        quality
    )

    if not quality["usable"]:

        print(
            "Epoch rejected by QC."
        )

        return

    # ----------------------------------------------
    # 8. NORMALIZATION
    # ----------------------------------------------

    epoch = zscore_normalize(
        epoch
    )

    print(
        "Normalized shape:",
        epoch.shape
    )

    # ----------------------------------------------
    # 9. CONVERT TO PYTORCH
    # ----------------------------------------------

    eeg_tensor = torch.from_numpy(
        epoch
    ).float()

    # ----------------------------------------------
    # 10. MOVE TO GPU
    # ----------------------------------------------

    eeg_tensor = eeg_tensor.to(
        device
    )

    print(
        "Tensor shape:",
        eeg_tensor.shape
    )

    print(
        "Tensor device:",
        eeg_tensor.device
    )

    # ----------------------------------------------
    # 11. GPU STFT / FOURIER
    # ----------------------------------------------

    frequencies, times, power = eeg_to_stft(
        eeg_tensor,
        sampling_rate=TARGET_RATE
    )

    # ----------------------------------------------
    # 12. LOG POWER
    # ----------------------------------------------

    spectrogram = log_power(
        power
    )

    # ----------------------------------------------
    # 13. OUTPUT
    # ----------------------------------------------

    print("\nSTFT RESULT")

    print(
        "Frequency bins:",
        len(frequencies)
    )

    print(
        "Time bins:",
        len(times)
    )

    print(
        "Spectrogram shape:",
        spectrogram.shape
    )

    print(
        "Spectrogram device:",
        spectrogram.device
    )

    print(
        "\nREAL EEG GPU PIPELINE SUCCESSFUL"
    )


# --------------------------------------------------
# RUN NATHAN
# --------------------------------------------------

process_real_edf(
    nathan_file,
    "Nathan"
)


# --------------------------------------------------
# RUN SUMIT
# --------------------------------------------------

process_real_edf(
    sumit_file,
    "Sumit"
)