import numpy as np
from scipy.signal import butter, sosfiltfilt


LOW_CUTOFF = 0.5
HIGH_CUTOFF = 30.0
FILTER_ORDER = 4


def bandpass_filter(
    eeg,
    sampling_rate,
    low_cutoff=LOW_CUTOFF,
    high_cutoff=HIGH_CUTOFF,
    order=FILTER_ORDER
):
    """
    Apply zero-phase Butterworth band-pass filtering.

    Parameters
    ----------
    eeg : numpy.ndarray
        EEG array with shape:
        (channels, samples)

    sampling_rate : int or float
        Sampling frequency in Hz.

    Returns
    -------
    filtered : numpy.ndarray
        Filtered EEG with the same shape as input.
    """

    eeg = np.asarray(
        eeg,
        dtype=np.float32
    )

    if eeg.ndim != 2:
        raise ValueError(
            "EEG must have shape "
            "(channels, samples)."
        )

    nyquist = sampling_rate / 2.0

    if high_cutoff >= nyquist:
        raise ValueError(
            f"High cutoff {high_cutoff} Hz must be "
            f"below Nyquist frequency {nyquist} Hz."
        )

    sos = butter(
        order,
        [
            low_cutoff / nyquist,
            high_cutoff / nyquist
        ],
        btype="bandpass",
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        eeg,
        axis=-1
    )

    return filtered.astype(
        np.float32
    )


if __name__ == "__main__":

    # Test signal
    sampling_rate = 256
    duration = 30

    samples = (
        sampling_rate * duration
    )

    time = (
        np.arange(samples)
        / sampling_rate
    )

    # Example 10 Hz signal
    signal = np.sin(
        2 * np.pi * 10 * time
    )

    eeg = np.stack(
        [signal] * 9
    )

    filtered = bandpass_filter(
        eeg,
        sampling_rate
    )

    print(
        "Filtering test successful."
    )

    print(
        "Input shape:",
        eeg.shape
    )

    print(
        "Output shape:",
        filtered.shape
    )

    print(
        "Band:",
        LOW_CUTOFF,
        "-",
        HIGH_CUTOFF,
        "Hz"
    )