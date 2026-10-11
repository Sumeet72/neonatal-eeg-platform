from pathlib import Path
import sys

import mne
import numpy as np
import pandas as pd
import torch

from src.preprocessing.channel_selection import select_common_channels
from src.preprocessing.filtering import bandpass_filter
from src.preprocessing.resampling import resample_eeg
from src.preprocessing.artifact_detection import check_eeg_quality
from src.preprocessing.normalization import zscore_normalize
from src.transforms.stft import eeg_to_stft, log_power
from src.inference import EEGInference


TARGET_RATE = 256

EPOCH_DURATION = 30

EXPECTED_SAMPLES = TARGET_RATE * EPOCH_DURATION


def preprocess_epoch(raw, start_second, end_second):
    """
    Preprocess one EEG epoch using the exact
    pipeline used during model training.
    """

    original_rate = float(raw.info["sfreq"])

    # --------------------------------------------------
    # 1. Crop requested epoch
    # --------------------------------------------------

    epoch_raw = raw.copy()

    epoch_raw.crop(
        tmin=start_second,
        tmax=end_second,
        include_tmax=False
    )

    epoch_raw.load_data()

    # --------------------------------------------------
    # 2. Select common EEG channels
    # --------------------------------------------------

    epoch_raw = select_common_channels(
        epoch_raw
    )

    eeg = epoch_raw.get_data()

    # --------------------------------------------------
    # 3. Band-pass filtering
    # --------------------------------------------------

    eeg = bandpass_filter(
        eeg,
        original_rate
    )

    # --------------------------------------------------
    # 4. Resampling
    # --------------------------------------------------

    eeg = resample_eeg(
        eeg,
        original_rate,
        TARGET_RATE
    )

    # --------------------------------------------------
    # 5. Force exactly 30 seconds
    # --------------------------------------------------

    if eeg.shape[1] < EXPECTED_SAMPLES:

        padded = np.zeros(
            (
                eeg.shape[0],
                EXPECTED_SAMPLES
            ),
            dtype=np.float32
        )

        padded[:, :eeg.shape[1]] = eeg

        eeg = padded

    elif eeg.shape[1] > EXPECTED_SAMPLES:

        eeg = eeg[
            :, :EXPECTED_SAMPLES
        ]

    # --------------------------------------------------
    # 6. Quality check
    # --------------------------------------------------

    quality = check_eeg_quality(
        eeg
    )

    # --------------------------------------------------
    # 7. Z-score normalization
    # --------------------------------------------------

    eeg = zscore_normalize(
        eeg
    )

    # --------------------------------------------------
    # 8. Convert to PyTorch
    # --------------------------------------------------

    eeg_tensor = torch.from_numpy(
        eeg
    ).float()

    # --------------------------------------------------
    # 9. STFT
    # --------------------------------------------------

    frequencies, times, power = eeg_to_stft(
        eeg_tensor,
        sampling_rate=TARGET_RATE
    )

    # --------------------------------------------------
    # 10. Log power
    # --------------------------------------------------

    spectrogram = log_power(
        power
    )

    # --------------------------------------------------
    # 11. Final shape check
    # --------------------------------------------------

    expected_shape = (
        9,
        129,
        59
    )

    if tuple(spectrogram.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected spectrogram shape. "
            f"Expected {expected_shape}, "
            f"got {tuple(spectrogram.shape)}"
        )

    return spectrogram, quality


def predict_edf(edf_path):

    edf_path = Path(edf_path)

    if not edf_path.exists():

        raise FileNotFoundError(
            f"EDF file not found: {edf_path}"
        )

    print("=" * 70)
    print("NEONATAL EEG SEIZURE PREDICTION")
    print("=" * 70)

    print(f"\nEDF file:")
    print(edf_path)

    # --------------------------------------------------
    # Load EDF
    # --------------------------------------------------

    print("\nLoading EDF...")

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=False,
        verbose=False
    )

    duration = float(
        raw.times[-1]
    )

    sampling_rate = float(
        raw.info["sfreq"]
    )

    print(
        f"Original sampling rate: "
        f"{sampling_rate:.2f} Hz"
    )

    print(
        f"Recording duration: "
        f"{duration:.2f} seconds"
    )

    print(
        f"Original channels: "
        f"{len(raw.ch_names)}"
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading ResNet18 model...")

    inference = EEGInference()

    print(
        f"Device: {inference.device}"
    )

    print(
        f"Threshold: {inference.threshold:.2f}"
    )

    # --------------------------------------------------
    # Determine number of complete epochs
    # --------------------------------------------------

    num_epochs = int(
        duration // EPOCH_DURATION
    )

    if num_epochs == 0:

        raise ValueError(
            "EDF recording is shorter than "
            "one 30-second epoch."
        )

    print(
        f"\nComplete 30-second epochs: "
        f"{num_epochs}"
    )

    print("\nProcessing epochs...")
    print("-" * 70)

    results = []

    for epoch_index in range(num_epochs):

        start_second = (
            epoch_index * EPOCH_DURATION
        )

        end_second = (
            start_second + EPOCH_DURATION
        )

        try:

            spectrogram, quality = preprocess_epoch(
                raw,
                start_second,
                end_second
            )

            prediction = inference.predict(
                spectrogram
            )[0]

            seizure_probability = (
                prediction["seizure_probability"]
            )

            predicted_label = (
                prediction["prediction"]
            )

            label_name = (
                "SEIZURE"
                if predicted_label == 1
                else "NORMAL"
            )


            print(
                f"Epoch {epoch_index:03d} | "
                f"{start_second:6.1f}-{end_second:6.1f}s | "
                f"Probability: "
                f"{seizure_probability:.4f} | "
                f"{label_name}"
            )

            results.append(
                {
                    "epoch": epoch_index,
                    "start_second": start_second,
                    "end_second": end_second,
                    "seizure_probability": float(seizure_probability),
                    "prediction": int(predicted_label),
                    "prediction_label": label_name,
                    "threshold": float(inference.threshold)
                }
            )
        except Exception as error:

            print(
                f"Epoch {epoch_index:03d} | "
                f"ERROR: {error}"
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)

    if len(results_df) == 0:

        print(
            "No epochs were successfully processed."
        )

        return

    total_epochs = len(
        results_df
    )

    seizure_epochs = int(
        (
            results_df["prediction"] == 1
        ).sum()
    )

    normal_epochs = (
        total_epochs -
        seizure_epochs
    )

    max_probability = float(
        results_df[
            "seizure_probability"
        ].max()
    )

    mean_probability = float(
        results_df[
            "seizure_probability"
        ].mean()
    )

    print(
        f"Total processed epochs: "
        f"{total_epochs}"
    )

    print(
        f"Predicted normal epochs: "
        f"{normal_epochs}"
    )

    print(
        f"Predicted seizure epochs: "
        f"{seizure_epochs}"
    )

    print(
        f"Maximum seizure probability: "
        f"{max_probability:.4f}"
    )

    print(
        f"Mean seizure probability: "
        f"{mean_probability:.4f}"
    )

    if seizure_epochs > 0:

        print(
            "\nPotential seizure epochs:"
        )

        seizure_rows = results_df[
            results_df["prediction"] == 1
        ]

        for _, row in seizure_rows.iterrows():

            print(
                f"  Epoch {int(row['epoch'])}: "
                f"{row['start_second']:.1f}-"
                f"{row['end_second']:.1f}s | "
                f"P(seizure)="
                f"{row['seizure_probability']:.4f}"
            )

    else:

        print(
            "\nNo epochs crossed the "
            f"{inference.threshold:.2f} seizure threshold."
        )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output_dir = Path(
        "results/inference"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir /
        f"{edf_path.stem}_predictions.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nResults saved to:"
    )

    print(
        output_file
    )

    print("=" * 70)


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            'python -m scripts.predict_edf '
            '"path\\to\\recording.edf"'
        )

        sys.exit(1)

    predict_edf(
        sys.argv[1]
    )