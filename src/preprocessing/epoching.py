import numpy as np


def create_epoch_labels(
    second_labels,
    epoch_seconds=30,
    stride_seconds=15,
    seizure_threshold=0.5
):
    """
    Convert second-by-second seizure labels into
    fixed-length epoch labels.

    Parameters
    ----------
    second_labels : array-like
        0/1 seizure label for every second.

    epoch_seconds : int
        Length of each EEG epoch in seconds.

    stride_seconds : int
        Distance between consecutive epochs.

    seizure_threshold : float
        Fraction of seizure-labelled seconds required
        to classify the complete epoch as seizure.

        Example:
        threshold = 0.5
        30-second epoch needs at least 15 seizure seconds.
    """

    labels = np.asarray(second_labels, dtype=np.uint8)

    epoch_labels = []
    start_times = []

    start = 0

    while start + epoch_seconds <= len(labels):

        end = start + epoch_seconds

        epoch = labels[start:end]

        seizure_ratio = epoch.mean()

        label = int(
            seizure_ratio >= seizure_threshold
        )

        epoch_labels.append(label)
        start_times.append(start)

        start += stride_seconds

    return np.asarray(epoch_labels), np.asarray(start_times)


if __name__ == "__main__":

    # Small test example

    test_labels = np.zeros(100, dtype=np.uint8)

    # Put seizure labels into part of the signal
    test_labels[10:30] = 1

    labels, starts = create_epoch_labels(
        test_labels,
        epoch_seconds=30,
        stride_seconds=15,
        seizure_threshold=0.5
    )

    print("Number of epochs:", len(labels))

    print("Epoch start times:", starts)

    print("Epoch labels:", labels)