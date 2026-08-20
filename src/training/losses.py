import torch
import torch.nn as nn


def get_class_weights(labels):
    """
    Calculate inverse-frequency class weights.

    labels:
        1D array/tensor containing 0 and 1.
    """

    labels = torch.as_tensor(
        labels,
        dtype=torch.long
    )

    counts = torch.bincount(
        labels,
        minlength=2
    ).float()

    weights = (
        len(labels)
        / (2.0 * counts.clamp(min=1))
    )

    return weights


def get_weighted_loss(labels, device):
    """
    Create weighted CrossEntropyLoss
    for seizure/normal imbalance.
    """

    weights = get_class_weights(
        labels
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    return criterion, weights


if __name__ == "__main__":

    labels = torch.tensor(
        [0] * 10 + [1] * 2
    )

    weights = get_class_weights(
        labels
    )

    print(
        "Labels:",
        len(labels)
    )

    print(
        "Class weights:",
        weights
    )

    print(
        "Loss function test successful."
    )