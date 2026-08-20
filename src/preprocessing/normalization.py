import numpy as np


def zscore_normalize(eeg, epsilon=1e-8):
    """
    Channel-wise z-score normalization.

    Input:
        EEG shape = (channels, samples)

    Each channel is normalized independently.
    """

    eeg = np.asarray(
        eeg,
        dtype=np.float32
    )

    if eeg.ndim != 2:
        raise ValueError(
            "EEG must have shape (channels, samples)."
        )

    mean = np.mean(
        eeg,
        axis=1,
        keepdims=True
    )

    std = np.std(
        eeg,
        axis=1,
        keepdims=True
    )

    normalized = (
        eeg - mean
    ) / (
        std + epsilon
    )

    return normalized.astype(
        np.float32
    )


if __name__ == "__main__":

    sampling_rate = 256
    duration = 30
    samples = sampling_rate * duration

    # Example EEG with different channel amplitudes
    eeg = np.random.randn(
        9,
        samples
    ).astype(np.float32)

    normalized = zscore_normalize(eeg)

    print(
        "Normalization test successful."
    )

    print(
        "Input shape:",
        eeg.shape
    )

    print(
        "Output shape:",
        normalized.shape
    )

    print(
        "Mean after normalization:",
        normalized.mean(axis=1)
    )

    print(
        "Std after normalization:",
        normalized.std(axis=1)
    )