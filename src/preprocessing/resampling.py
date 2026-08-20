import numpy as np
from scipy.signal import resample_poly


TARGET_SAMPLING_RATE = 256


def resample_eeg(
    eeg,
    original_rate,
    target_rate=TARGET_SAMPLING_RATE
):
    """
    Resample EEG to a common sampling rate.

    Parameters
    ----------
    eeg : numpy.ndarray
        EEG with shape:
        (channels, samples)

    original_rate : int or float
        Original sampling frequency.

    target_rate : int
        Target sampling frequency.

    Returns
    -------
    resampled : numpy.ndarray
        Resampled EEG.
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

    if original_rate <= 0:
        raise ValueError(
            "Original sampling rate must be positive."
        )

    if target_rate <= 0:
        raise ValueError(
            "Target sampling rate must be positive."
        )

    if original_rate == target_rate:
        return eeg.copy()

    # Convert rates to integers for resample_poly.
    original_rate = int(original_rate)
    target_rate = int(target_rate)

    # Greatest common divisor simplifies
    # the up/down sampling factors.
    gcd = np.gcd(
        original_rate,
        target_rate
    )

    up = target_rate // gcd
    down = original_rate // gcd

    resampled = resample_poly(
        eeg,
        up=up,
        down=down,
        axis=-1
    )

    return resampled.astype(
        np.float32
    )


if __name__ == "__main__":

    # Test 200 Hz → 256 Hz

    original_rate = 200
    target_rate = 256

    duration = 30

    samples = (
        original_rate * duration
    )

    time = (
        np.arange(samples)
        / original_rate
    )

    signal = np.sin(
        2 * np.pi * 10 * time
    )

    eeg = np.stack(
        [signal] * 9
    )

    resampled = resample_eeg(
        eeg,
        original_rate,
        target_rate
    )

    print(
        "Resampling test successful."
    )

    print(
        "Original rate:",
        original_rate,
        "Hz"
    )

    print(
        "Target rate:",
        target_rate,
        "Hz"
    )

    print(
        "Input shape:",
        eeg.shape
    )

    print(
        "Output shape:",
        resampled.shape
    )

    print(
        "Expected samples:",
        target_rate * duration
    )