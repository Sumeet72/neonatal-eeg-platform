import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

PREDICTION_FILE = Path(
    "results/ResNet18_evaluation/ResNet18_all_LOSO_predictions.csv"
)

OUTPUT_DIR = Path("results/ResNet18_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.40

df = pd.read_csv(PREDICTION_FILE)

print("=" * 70)
print("RESNET-18 SUBJECT-WISE ANALYSIS")
print("=" * 70)

rows = []

for subject, group in df.groupby("subject_id"):

    y_true = group["true_label"].astype(int).values
    probability = group["seizure_probability"].astype(float).values

    y_pred = (probability >= THRESHOLD).astype(int)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    try:
        auc = roc_auc_score(y_true, probability)
    except ValueError:
        auc = np.nan

    rows.append({
        "subject_id": subject,
        "epochs": len(group),
        "normal_epochs": int(np.sum(y_true == 0)),
        "seizure_epochs": int(np.sum(y_true == 1)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    })

subject_df = pd.DataFrame(rows)

subject_df = subject_df.sort_values("subject_id")

output_file = OUTPUT_DIR / "ResNet18_subject_wise_results.csv"

subject_df.to_csv(output_file, index=False)

print()
print(f"Subjects analyzed: {len(subject_df)}")
print(f"Saved to: {output_file}")

print()
print("=" * 70)
print("SUBJECT-WISE SUMMARY")
print("=" * 70)

print(
    f"Mean sensitivity: "
    f"{subject_df['sensitivity'].mean():.4f}"
)

print(
    f"Median sensitivity: "
    f"{subject_df['sensitivity'].median():.4f}"
)

print(
    f"Mean specificity: "
    f"{subject_df['specificity'].mean():.4f}"
)

print(
    f"Median specificity: "
    f"{subject_df['specificity'].median():.4f}"
)

print(
    f"Mean F1: "
    f"{subject_df['f1'].mean():.4f}"
)

print(
    f"Median F1: "
    f"{subject_df['f1'].median():.4f}"
)

print()
print("Subjects containing seizure epochs:")

positive_subjects = subject_df[
    subject_df["seizure_epochs"] > 0
]

print(
    f"{len(positive_subjects)} / "
    f"{len(subject_df)} subjects"
)

print()
print("Top 10 subjects by sensitivity:")

print(
    positive_subjects[
        [
            "subject_id",
            "seizure_epochs",
            "sensitivity",
            "specificity",
            "f1",
            "roc_auc"
        ]
    ]
    .sort_values("sensitivity", ascending=False)
    .head(10)
    .to_string(index=False)
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
