import os
import glob
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PREDICTION_DIR = "results/ResNet18_loso_predictions"
OUTPUT_DIR = "results/ResNet18_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD ALL LOSO PREDICTIONS
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(PREDICTION_DIR, "*.csv")
    )
)

print("=" * 70)
print("RESNET-18 LOSO EVALUATION")
print("=" * 70)

print(f"Prediction files found: {len(files)}")

if len(files) != 132:
    print(
        f"WARNING: Expected 132 prediction files, "
        f"but found {len(files)}."
    )


all_predictions = []

for file in files:

    df = pd.read_csv(file)

    required_columns = [
        "subject_id",
        "epoch_id",
        "true_label",
        "predicted_label",
        "seizure_probability",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print(
            f"Skipping {file} - missing columns: {missing}"
        )
        continue

    all_predictions.append(df)


# ============================================================
# COMBINE
# ============================================================

combined = pd.concat(
    all_predictions,
    ignore_index=True
)

print(f"Total predictions: {len(combined)}")
print(
    f"Unique subjects: "
    f"{combined['subject_id'].nunique()}"
)

print(
    f"Normal epochs: "
    f"{(combined['true_label'] == 0).sum()}"
)

print(
    f"Seizure epochs: "
    f"{(combined['true_label'] == 1).sum()}"
)


# Save combined predictions
combined_file = os.path.join(
    OUTPUT_DIR,
    "ResNet18_all_LOSO_predictions.csv"
)

combined.to_csv(
    combined_file,
    index=False
)

print(
    f"\nCombined predictions saved to:\n"
    f"{combined_file}"
)


# ============================================================
# VARIABLES
# ============================================================

y_true = combined["true_label"].astype(int).values

y_pred = combined["predicted_label"].astype(int).values

y_prob = combined["seizure_probability"].astype(float).values


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

sensitivity = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

specificity = (
    confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )[0, 0]
    /
    (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )[0, 0]
        +
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )[0, 1]
    )
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

pr_auc = average_precision_score(
    y_true,
    y_prob
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


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL RESNET-18 PERFORMANCE")
print("=" * 70)

print(f"Accuracy:     {accuracy:.4f}")
print(f"Precision:    {precision:.4f}")
print(f"Sensitivity:  {sensitivity:.4f}")
print(f"Specificity:  {specificity:.4f}")
print(f"F1-score:     {f1:.4f}")
print(f"ROC-AUC:      {roc_auc:.4f}")
print(f"PR-AUC:       {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\n")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "Normal",
        "Seizure"
    ],
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1-score",
            "ROC-AUC",
            "PR-AUC",
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives",
            "Total Epochs",
            "Total Subjects",
        ],
        "Value": [
            accuracy,
            precision,
            sensitivity,
            specificity,
            f1,
            roc_auc,
            pr_auc,
            tn,
            fp,
            fn,
            tp,
            len(combined),
            combined["subject_id"].nunique(),
        ],
    }
)

metrics_file = os.path.join(
    OUTPUT_DIR,
    "ResNet18_overall_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_true,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ResNet-18 (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ResNet-18 ROC Curve - Combined LOSO"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ResNet18_ROC_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_true,
        y_prob
    )
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision")

plt.title(
    "ResNet-18 Precision-Recall Curve"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ResNet18_PR_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "ResNet-18 Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    ["Normal", "Seizure"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Seizure"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ResNet18_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = np.arange(
    0.05,
    0.96,
    0.05
)

threshold_results = []

for threshold in thresholds:

    threshold_pred = (
        y_prob >= threshold
    ).astype(int)

    cm_t = confusion_matrix(
        y_true,
        threshold_pred,
        labels=[0, 1]
    )

    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

    sensitivity_t = (
        tp_t / (tp_t + fn_t)
        if (tp_t + fn_t) > 0
        else 0
    )

    specificity_t = (
        tn_t / (tn_t + fp_t)
        if (tn_t + fp_t) > 0
        else 0
    )

    precision_t = precision_score(
        y_true,
        threshold_pred,
        zero_division=0
    )

    f1_t = f1_score(
        y_true,
        threshold_pred,
        zero_division=0
    )

    accuracy_t = accuracy_score(
        y_true,
        threshold_pred
    )

    threshold_results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy_t,
            "precision": precision_t,
            "sensitivity": sensitivity_t,
            "specificity": specificity_t,
            "f1": f1_t,
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)

threshold_file = os.path.join(
    OUTPUT_DIR,
    "ResNet18_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False
)


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

print("\n")
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print(
    f"Best tested threshold by F1: "
    f"{best_row['threshold']:.2f}"
)

print(
    f"F1: {best_row['f1']:.4f}"
)

print(
    f"Sensitivity: "
    f"{best_row['sensitivity']:.4f}"
)

print(
    f"Specificity: "
    f"{best_row['specificity']:.4f}"
)

print(
    f"Precision: "
    f"{best_row['precision']:.4f}"
)


# ============================================================
# SENSITIVITY / SPECIFICITY VS THRESHOLD
# ============================================================

plt.figure(figsize=(8, 6))

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

plt.xlabel("Threshold")
plt.ylabel("Score")

plt.title(
    "Sensitivity and Specificity vs Threshold"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ResNet18_sensitivity_specificity_threshold.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# F1 VS THRESHOLD
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1"],
    marker="o"
)

plt.xlabel("Threshold")
plt.ylabel("F1-score")

plt.title(
    "F1-score vs Threshold"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ResNet18_F1_threshold.png"
    ),
    dpi=300
)

plt.close()


print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Output directory:\n"
    f"{OUTPUT_DIR}"
)