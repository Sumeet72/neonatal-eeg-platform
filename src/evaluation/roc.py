import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)


def calculate_roc_auc(
    y_true,
    y_probability
):

    y_true = np.asarray(
        y_true
    )

    y_probability = np.asarray(
        y_probability
    )

    # ROC-AUC requires both classes
    if len(
        np.unique(y_true)
    ) < 2:

        return {
            "roc_auc": None,
            "fpr": np.array([]),
            "tpr": np.array([]),
            "thresholds": np.array([])
        }

    auc = roc_auc_score(
        y_true,
        y_probability
    )

    fpr, tpr, thresholds = (
        roc_curve(
            y_true,
            y_probability
        )
    )

    return {
        "roc_auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }


if __name__ == "__main__":

    y_true = [
        0, 0, 0, 0,
        1, 1, 1, 1
    ]

    y_probability = [
        0.10,
        0.20,
        0.40,
        0.30,
        0.70,
        0.80,
        0.60,
        0.90
    ]

    results = calculate_roc_auc(
        y_true,
        y_probability
    )

    print(
        "ROC-AUC:",
        f"{results['roc_auc']:.4f}"
    )

    print(
        "FPR points:",
        len(results["fpr"])
    )

    print(
        "TPR points:",
        len(results["tpr"])
    )