import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_confusion_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(y_true)

    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity
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

    results = calculate_confusion_metrics(
        y_true,
        y_pred
    )

    print("Confusion Metrics:")

    for key, value in results.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )