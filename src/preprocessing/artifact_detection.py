import numpy as np


def check_eeg_quality(
    eeg,
    max_nan_ratio=0.01,
    max_flat_channel_ratio=0.5
):
    """
    Basic EEG quality-control checks.

    EEG shape:
        (channels, samples)

    The flat-channel check is relative to the
    median channel standard deviation so that
    datasets with different physical amplitude
    scales can be handled more safely.
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

    # -----------------------------------------
    # 1. NaN / infinite values
    # -----------------------------------------

    invalid_values = ~np.isfinite(eeg)

    invalid_ratio = invalid_values.mean()

    if invalid_ratio > max_nan_ratio:

        return {
            "usable": False,
            "reason": "too_many_invalid_values",
            "invalid_ratio": float(
                invalid_ratio
            )
        }

    # -----------------------------------------
    # 2. Replace invalid values temporarily
    # -----------------------------------------

    clean_eeg = np.nan_to_num(
        eeg,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # -----------------------------------------
    # 3. Channel standard deviations
    # -----------------------------------------

    channel_std = np.std(
        clean_eeg,
        axis=1
    )

    median_std = np.median(
        channel_std
    )

    # -----------------------------------------
    # 4. Relative flat-channel detection
    # -----------------------------------------

    if median_std == 0:

        return {
            "usable": False,
            "reason": "all_channels_flat",
            "invalid_ratio": float(
                invalid_ratio
            )
        }

    relative_std = (
        channel_std / median_std
    )

    flat_channels = np.where(
        relative_std < max_flat_channel_ratio
    )[0]

    # -----------------------------------------
    # 5. Too many flat channels
    # -----------------------------------------

    flat_ratio = (
        len(flat_channels)
        / eeg.shape[0]
    )

    too_many_flat_channels = (
        flat_ratio > 0.5
    )

    # -----------------------------------------
    # 6. Final decision
    # -----------------------------------------

    usable = not too_many_flat_channels

    reasons = []

    if too_many_flat_channels:
        reasons.append(
            "too_many_flat_channels"
        )

    if not reasons:
        reasons.append("OK")

    return {
        "usable": usable,
        "reason": ";".join(reasons),
        "invalid_ratio": float(
            invalid_ratio
        ),
        "channel_std": channel_std.tolist(),
        "median_std": float(
            median_std
        ),
        "flat_channels": flat_channels.tolist()
    }


if __name__ == "__main__":

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
        "Median channel std:",
        result["median_std"]
    )

    print(
        "Flat channels:",
        result["flat_channels"]
    )