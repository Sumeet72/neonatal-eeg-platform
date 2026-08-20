from pathlib import Path
import mne
import numpy as np


def extract_epochs(
    edf_path,
    second_labels,
    epoch_seconds=30,
    stride_seconds=15,
    seizure_threshold=0.5
):
    """
    Read one EDF file and extract fixed-length EEG epochs.

    Returns
    -------
    epochs : list of dict
        Each dictionary contains:
        - eeg
        - start_second
        - end_second
        - label
    """

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False
    )

    data = raw.get_data()

    sfreq = raw.info["sfreq"]

    duration = raw.times[-1]

    usable_seconds = min(
        int(duration),
        len(second_labels)
    )

    epochs = []

    start = 0

    while start + epoch_seconds <= usable_seconds:

        end = start + epoch_seconds

        start_sample = int(start * sfreq)
        end_sample = int(end * sfreq)

        eeg = data[:, start_sample:end_sample]

        annotation_segment = np.asarray(
            second_labels[start:end],
            dtype=np.uint8
        )

        seizure_ratio = annotation_segment.mean()

        label = int(
            seizure_ratio >= seizure_threshold
        )

        epochs.append(
            {
                "eeg": eeg,
                "start_second": start,
                "end_second": end,
                "label": label
            }
        )

        start += stride_seconds

    return epochs


if __name__ == "__main__":

    print("EDF epoch extractor loaded successfully.")