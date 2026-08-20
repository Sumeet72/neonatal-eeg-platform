import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }


if __name__ == "__main__":

    y_true = [
        0, 0, 0, 0,
        1, 1, 1, 1
    ]

    y_pred = [
        0, 0, 1, 0,
        1, 1, 0, 1
    ]

    results = calculate_metrics(
        y_true,
        y_pred
    )

    print(
        "Metrics:"
    )

    for key, value in results.items():

        print(
            f"{key}: {value:.4f}"
        )