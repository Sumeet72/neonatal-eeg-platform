import numpy as np


def check_eeg_quality(
    eeg,
    min_amplitude=-500.0,
    max_amplitude=500.0,
    flat_std_threshold=1e-6,
    max_nan_ratio=0.01
):
    """
    Basic EEG quality-control checks.

    Parameters
    ----------
    eeg : numpy.ndarray
        Shape: (channels, samples)

    Returns
    -------
    result : dict
        Quality information and usability flag.
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

    # NaN / infinite values
    invalid_values = ~np.isfinite(eeg)

    invalid_ratio = invalid_values.mean()

    has_too_many_invalid = (
        invalid_ratio > max_nan_ratio
    )

    # Replace invalid values temporarily for
    # amplitude/statistical calculations.
    clean_values = eeg[
        np.isfinite(eeg)
    ]

    if len(clean_values) == 0:
        return {
            "usable": False,
            "reason": "No finite EEG samples"
        }

    # Extreme amplitude check
    min_value = float(
        clean_values.min()
    )

    max_value = float(
        clean_values.max()
    )

    extreme_amplitude = (
        min_value < min_amplitude
        or max_value > max_amplitude
    )

    # Flat-channel check
    channel_std = np.std(
        np.nan_to_num(eeg),
        axis=1
    )

    flat_channels = np.where(
        channel_std < flat_std_threshold
    )[0]

    has_flat_channel = (
        len(flat_channels) > 0
    )

    # Overall quality decision
    usable = not (
        has_too_many_invalid
        or extreme_amplitude
        or has_flat_channel
    )

    reasons = []

    if has_too_many_invalid:
        reasons.append(
            "too_many_invalid_values"
        )

    if extreme_amplitude:
        reasons.append(
            "extreme_amplitude"
        )

    if has_flat_channel:
        reasons.append(
            "flat_channel"
        )

    if not reasons:
        reasons.append("OK")

    return {
        "usable": usable,
        "reason": ";".join(reasons),
        "invalid_ratio": float(
            invalid_ratio
        ),
        "min_amplitude": min_value,
        "max_amplitude": max_value,
        "flat_channels": flat_channels.tolist()
    }


if __name__ == "__main__":

    # Test with a clean synthetic EEG signal

    sampling_rate = 256
    duration = 30

    samples = (
        sampling_rate * duration
    )

    time = (
        np.arange(samples)
        / sampling_rate
    )

    signal = np.sin(
        2 * np.pi * 10 * time
    )

    eeg = np.stack(
        [signal] * 9
    )

    result = check_eeg_quality(
        eeg
    )

    print(
        "Artifact/QC test successful."
    )

    print(
        "Usable:",
        result["usable"]
    )

    print(
        "Reason:",
        result["reason"]
    )

    print(
        "Amplitude:",
        result["min_amplitude"],
        "to",
        result["max_amplitude"]
    )