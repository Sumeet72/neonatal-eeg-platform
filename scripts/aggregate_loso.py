import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


RESULTS_FILE = "results/loso_results.csv"
PREDICTION_DIR = "results/loso_predictions"


# ============================================================
# LOAD FOLD RESULTS
# ============================================================

df = pd.read_csv(
    RESULTS_FILE
)

print("============================================================")
print("LOSO AGGREGATION")
print("============================================================")

print(
    "Total folds:",
    len(df)
)


# ============================================================
# FOLD-LEVEL STATISTICS
# ============================================================

metrics = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc"
]


print("\n============================================================")
print("FOLD-LEVEL RESULTS")
print("============================================================")

for metric in metrics:

    values = pd.to_numeric(
        df[metric],
        errors="coerce"
    ).dropna()

    mean = values.mean()
    std = values.std()

    print(
        f"{metric:12s}: "
        f"{mean:.4f} ± {std:.4f}"
    )

    print(
        f"{'':12s}  "
        f"Valid folds: {len(values)}"
    )


# ============================================================
# TRAINING TIME
# ============================================================

training_time = pd.to_numeric(
    df["training_time_minutes"],
    errors="coerce"
).dropna()


print("\n============================================================")
print("TRAINING TIME")
print("============================================================")

print(
    "Total training time:",
    f"{training_time.sum() / 60:.2f} hours"
)

print(
    "Average fold time:",
    f"{training_time.mean():.2f} minutes"
)

print(
    "Median fold time:",
    f"{training_time.median():.2f} minutes"
)

print(
    "Minimum fold time:",
    f"{training_time.min():.2f} minutes"
)

print(
    "Maximum fold time:",
    f"{training_time.max():.2f} minutes"
)


# ============================================================
# LOAD ALL PREDICTIONS
# ============================================================

prediction_files = [
    f
    for f in os.listdir(PREDICTION_DIR)
    if f.endswith(".csv")
]


print("\n============================================================")
print("PREDICTIONS")
print("============================================================")

print(
    "Prediction files:",
    len(prediction_files)
)


all_predictions = []


for filename in prediction_files:

    path = os.path.join(
        PREDICTION_DIR,
        filename
    )

    prediction_df = pd.read_csv(
        path
    )

    all_predictions.append(
        prediction_df
    )


predictions = pd.concat(
    all_predictions,
    ignore_index=True
)


print(
    "Total predictions:",
    len(predictions)
)


# ============================================================
# OVERALL SAMPLE-LEVEL METRICS
# ============================================================

y_true = predictions[
    "true_label"
].to_numpy()

y_pred = predictions[
    "predicted_label"
].to_numpy()

y_probability = predictions[
    "seizure_probability"
].to_numpy()


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

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()


specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


try:

    roc_auc = roc_auc_score(
        y_true,
        y_probability
    )

except ValueError:

    roc_auc = None


# ============================================================
# OVERALL REPORT
# ============================================================

print("\n============================================================")
print("OVERALL LOSO PERFORMANCE")
print("============================================================")

print(
    "Accuracy:     ",
    f"{accuracy:.4f}"
)

print(
    "Precision:    ",
    f"{precision:.4f}"
)

print(
    "Sensitivity:  ",
    f"{sensitivity:.4f}"
)

print(
    "Specificity:  ",
    f"{specificity:.4f}"
)

print(
    "F1-score:     ",
    f"{f1:.4f}"
)

print(
    "ROC-AUC:      ",
    (
        f"{roc_auc:.4f}"
        if roc_auc is not None
        else "N/A"
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n============================================================")
print("OVERALL CONFUSION MATRIX")
print("============================================================")

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


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n============================================================")
print("OVERALL CLASS DISTRIBUTION")
print("============================================================")

unique, counts = np.unique(
    y_true,
    return_counts=True
)

for label, count in zip(
    unique,
    counts
):

    print(
        f"Class {label}:",
        count
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "accuracy": accuracy,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "roc_auc": roc_auc,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total_samples": len(y_true),
            "folds": len(df)
        }
    ]
)


summary_file = (
    "results/loso_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


print(
    "\nSummary saved:",
    summary_file
)

print(
    "\n============================================================"
)

print(
    "LOSO AGGREGATION COMPLETE"
)

print(
    "============================================================"
)