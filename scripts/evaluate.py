import os

import numpy as np
import pandas as pd

from src.evaluation.report import (
    generate_report,
    print_report
)


# ============================================================
# SETTINGS
# ============================================================

PREDICTION_DIR = (
    "results/loso_predictions"
)

THRESHOLD = 0.40


# ============================================================
# LOAD LOSO PREDICTIONS
# ============================================================

files = [
    file
    for file in os.listdir(
        PREDICTION_DIR
    )
    if file.endswith(".csv")
]


frames = []

for file in files:

    path = os.path.join(
        PREDICTION_DIR,
        file
    )

    frames.append(
        pd.read_csv(path)
    )


df = pd.concat(
    frames,
    ignore_index=True
)


print(
    "============================================================"
)

print(
    "FINAL CNN EVALUATION"
)

print(
    "============================================================"
)

print(
    "Prediction files:",
    len(files)
)

print(
    "Total samples:",
    len(df)
)


# ============================================================
# DATA
# ============================================================

y_true = df[
    "true_label"
].to_numpy()

probability = df[
    "seizure_probability"
].to_numpy()


# ============================================================
# STANDARD 0.50 THRESHOLD
# ============================================================

y_pred_50 = (
    probability >= 0.50
).astype(int)


report_50 = generate_report(
    y_true,
    y_pred_50,
    probability
)


print(
    "\n============================================================"
)

print(
    "THRESHOLD = 0.50"
)

print(
    "============================================================"
)

print_report(
    report_50
)


# ============================================================
# OPTIMIZED 0.40 THRESHOLD
# ============================================================

y_pred_40 = (
    probability >= THRESHOLD
).astype(int)


report_40 = generate_report(
    y_true,
    y_pred_40,
    probability
)


print(
    "\n============================================================"
)

print(
    "THRESHOLD = 0.40"
)

print(
    "============================================================"
)

print_report(
    report_40
)


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "threshold": 0.50,
            **report_50
        },
        {
            "threshold": THRESHOLD,
            **report_40
        }
    ]
)


output_file = (
    "results/final_evaluation.csv"
)


summary.to_csv(
    output_file,
    index=False
)


print(
    "\nFinal evaluation saved:",
    output_file
)


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print(
    "\n============================================================"
)

print(
    "FINAL OPERATING POINT"
)

print(
    "============================================================"
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Accuracy:",
    f"{report_40['accuracy']:.4f}"
)

print(
    "Sensitivity:",
    f"{report_40['sensitivity']:.4f}"
)

print(
    "Specificity:",
    f"{report_40['specificity']:.4f}"
)

print(
    "Precision:",
    f"{report_40['precision']:.4f}"
)

print(
    "F1-score:",
    f"{report_40['f1']:.4f}"
)

if report_40["roc_auc"] is not None:

    print(
        "ROC-AUC:",
        f"{report_40['roc_auc']:.4f}"
    )