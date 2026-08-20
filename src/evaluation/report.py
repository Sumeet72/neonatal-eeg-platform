from src.evaluation.metrics import (
    calculate_metrics
)

from src.evaluation.confusion_metrics import (
    calculate_confusion_metrics
)

from src.evaluation.roc import (
    calculate_roc_auc
)


def generate_report(
    y_true,
    y_pred,
    y_probability
):

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    confusion = (
        calculate_confusion_metrics(
            y_true,
            y_pred
        )
    )

    roc = calculate_roc_auc(
        y_true,
        y_probability
    )

    report = {}

    report.update(
        metrics
    )

    report.update(
        confusion
    )

    report["roc_auc"] = (
        roc["roc_auc"]
    )

    return report


def print_report(
    report
):

    print(
        "\n============================================================"
    )

    print(
        "EEG EVALUATION REPORT"
    )

    print(
        "============================================================"
    )

    print(
        f"Accuracy:     "
        f"{report['accuracy']:.4f}"
    )

    print(
        f"Precision:    "
        f"{report['precision']:.4f}"
    )

    print(
        f"Recall:       "
        f"{report['recall']:.4f}"
    )

    print(
        f"F1-score:     "
        f"{report['f1']:.4f}"
    )

    print(
        f"Sensitivity:  "
        f"{report['sensitivity']:.4f}"
    )

    print(
        f"Specificity:  "
        f"{report['specificity']:.4f}"
    )

    if report["roc_auc"] is not None:

        print(
            f"ROC-AUC:      "
            f"{report['roc_auc']:.4f}"
        )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"TN: {report['TN']}"
    )

    print(
        f"FP: {report['FP']}"
    )

    print(
        f"FN: {report['FN']}"
    )

    print(
        f"TP: {report['TP']}"
    )


if __name__ == "__main__":

    y_true = [
        0, 0, 0, 0,
        1, 1, 1, 1
    ]

    y_pred = [
        0, 0, 1, 0,
        1, 1, 0, 1
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

    report = generate_report(
        y_true,
        y_pred,
        y_probability
    )

    print_report(
        report
    )