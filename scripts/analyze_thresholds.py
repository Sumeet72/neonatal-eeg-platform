import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)


PREDICTION_FILE = "results/loso_predictions"


# ============================================================
# LOAD ALL PREDICTIONS
# ============================================================

import os

files = [
    f
    for f in os.listdir(PREDICTION_FILE)
    if f.endswith(".csv")
]

frames = []

for file in files:

    path = os.path.join(
        PREDICTION_FILE,
        file
    )

    frames.append(
        pd.read_csv(path)
    )

df = pd.concat(
    frames,
    ignore_index=True
)

print("============================================================")
print("THRESHOLD ANALYSIS")
print("============================================================")

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
# BASELINE ROC / PR
# ============================================================

roc_auc = roc_auc_score(
    y_true,
    probability
)

pr_auc = average_precision_score(
    y_true,
    probability
)

print(
    "\nROC-AUC:",
    f"{roc_auc:.4f}"
)

print(
    "PR-AUC:",
    f"{pr_auc:.4f}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = np.arange(
    0.10,
    0.56,
    0.05
)

results = []


print("\n============================================================")
print("THRESHOLD RESULTS")
print("============================================================")

print(
    "Threshold | Accuracy | Precision | Sensitivity | "
    "Specificity | F1"
)

print(
    "------------------------------------------------------------"
)


for threshold in thresholds:

    y_pred = (
        probability >= threshold
    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()


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
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    )


    print(
        f"{threshold:9.2f} | "
        f"{accuracy:8.4f} | "
        f"{precision:9.4f} | "
        f"{sensitivity:11.4f} | "
        f"{specificity:11.4f} | "
        f"{f1:6.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

output_file = (
    "results/threshold_analysis.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# BEST F1
# ============================================================

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]


print("\n============================================================")
print("BEST F1 THRESHOLD")
print("============================================================")

print(
    "Threshold:",
    f"{best_f1['threshold']:.2f}"
)

print(
    "Accuracy:",
    f"{best_f1['accuracy']:.4f}"
)

print(
    "Precision:",
    f"{best_f1['precision']:.4f}"
)

print(
    "Sensitivity:",
    f"{best_f1['sensitivity']:.4f}"
)

print(
    "Specificity:",
    f"{best_f1['specificity']:.4f}"
)

print(
    "F1:",
    f"{best_f1['f1']:.4f}"
)


print(
    "\nResults saved:",
    output_file
)

print(
    "\n============================================================"
)

print(
    "THRESHOLD ANALYSIS COMPLETE"
)

print(
    "============================================================"
)