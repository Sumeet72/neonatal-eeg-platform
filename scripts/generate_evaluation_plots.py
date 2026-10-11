import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# SETTINGS
# ============================================================

PREDICTION_DIR = "results/loso_predictions"
SUBJECT_FILE = "results/subject_analysis.csv"

OUTPUT_DIR = "results/figures"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD ALL PREDICTIONS
# ============================================================

files = [
    f
    for f in os.listdir(PREDICTION_DIR)
    if f.endswith(".csv")
]

frames = []

for filename in files:

    path = os.path.join(
        PREDICTION_DIR,
        filename
    )

    frames.append(
        pd.read_csv(path)
    )


df = pd.concat(
    frames,
    ignore_index=True
)


y_true = df[
    "true_label"
].to_numpy()

y_probability = df[
    "seizure_probability"
].to_numpy()

y_pred = df[
    "predicted_label"
].to_numpy()


print("============================================================")
print("EVALUATION PLOTS")
print("============================================================")

print(
    "Prediction files:",
    len(files)
)

print(
    "Total predictions:",
    len(df)
)


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_true,
    y_probability
)

roc_auc = roc_auc_score(
    y_true,
    y_probability
)


plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "LOSO ROC Curve"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/roc_curve.png",
    dpi=300
)

plt.close()


print(
    "Saved: roc_curve.png"
)


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision, recall, _ = precision_recall_curve(
    y_true,
    y_probability
)

pr_auc = average_precision_score(
    y_true,
    y_probability
)


plt.figure()

plt.plot(
    recall,
    precision,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel(
    "Recall / Sensitivity"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "LOSO Precision-Recall Curve"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/precision_recall_curve.png",
    dpi=300
)

plt.close()


print(
    "Saved: precision_recall_curve.png"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


plt.figure()

plt.imshow(
    cm
)

plt.xticks(
    [0, 1],
    ["Normal", "Seizure"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Seizure"]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.title(
    "LOSO Confusion Matrix"
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/confusion_matrix.png",
    dpi=300
)

plt.close()


print(
    "Saved: confusion_matrix.png"
)


# ============================================================
# THRESHOLD CURVES
# ============================================================

threshold_file = (
    "results/threshold_analysis.csv"
)

threshold_df = pd.read_csv(
    threshold_file
)


plt.figure()

plt.plot(
    threshold_df["threshold"],
    threshold_df["sensitivity"],
    marker="o",
    label="Sensitivity"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["specificity"],
    marker="o",
    label="Specificity"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Sensitivity vs Specificity by Threshold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/threshold_sensitivity_specificity.png",
    dpi=300
)

plt.close()


print(
    "Saved: threshold_sensitivity_specificity.png"
)


# ============================================================
# F1 VS THRESHOLD
# ============================================================

plt.figure()

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1"],
    marker="o"
)

best_index = threshold_df[
    "f1"
].idxmax()

best_threshold = threshold_df.loc[
    best_index,
    "threshold"
]

best_f1 = threshold_df.loc[
    best_index,
    "f1"
]


plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best F1 = {best_threshold:.2f}"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "F1-score"
)

plt.title(
    "F1-score vs Classification Threshold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/f1_threshold.png",
    dpi=300
)

plt.close()


print(
    "Saved: f1_threshold.png"
)


# ============================================================
# SUBJECT SENSITIVITY DISTRIBUTION
# ============================================================

subject_df = pd.read_csv(
    SUBJECT_FILE
)

sensitivity = subject_df[
    "sensitivity"
].to_numpy()


plt.figure()

plt.hist(
    sensitivity,
    bins=20
)

plt.xlabel(
    "Subject-level Sensitivity"
)

plt.ylabel(
    "Number of Subjects"
)

plt.title(
    "Distribution of Subject-level Sensitivity"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/subject_sensitivity_distribution.png",
    dpi=300
)

plt.close()


print(
    "Saved: subject_sensitivity_distribution.png"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================================")
print("PLOT SUMMARY")
print("============================================================")

print(
    "ROC-AUC:",
    f"{roc_auc:.4f}"
)

print(
    "PR-AUC:",
    f"{pr_auc:.4f}"
)

print(
    "TN:",
    tn
)

print(
    "FP:",
    fp
)

print(
    "FN:",
    fn
)

print(
    "TP:",
    tp
)

print(
    "Best tested threshold:",
    f"{best_threshold:.2f}"
)

print(
    "Best tested F1:",
    f"{best_f1:.4f}"
)

print(
    "\nFigures directory:",
    OUTPUT_DIR
)

print(
    "\n============================================================"
)

print(
    "EVALUATION PLOTS COMPLETE"
)

print(
    "============================================================"
)