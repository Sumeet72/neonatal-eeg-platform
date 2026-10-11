import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


PREDICTION_DIR = "results/loso_predictions"

OUTPUT_FILE = "results/subject_analysis.csv"


# ============================================================
# LOAD PREDICTIONS
# ============================================================

files = [
    f
    for f in os.listdir(PREDICTION_DIR)
    if f.endswith(".csv")
]

print("============================================================")
print("SUBJECT-WISE LOSO ANALYSIS")
print("============================================================")

print(
    "Prediction files:",
    len(files)
)


# ============================================================
# ANALYZE EACH SUBJECT
# ============================================================

results = []


for filename in files:

    path = os.path.join(
        PREDICTION_DIR,
        filename
    )

    df = pd.read_csv(
        path
    )

    subject = str(
        df["subject_id"].iloc[0]
    )

    y_true = df[
        "true_label"
    ].to_numpy()

    y_pred = df[
        "predicted_label"
    ].to_numpy()

    probability = df[
        "seizure_probability"
    ].to_numpy()


    total_epochs = len(
        df
    )

    seizure_epochs = int(
        np.sum(y_true == 1)
    )

    normal_epochs = int(
        np.sum(y_true == 0)
    )


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


    tn = np.sum(
        (y_true == 0) &
        (y_pred == 0)
    )

    fp = np.sum(
        (y_true == 0) &
        (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1) &
        (y_pred == 0)
    )

    tp = np.sum(
        (y_true == 1) &
        (y_pred == 1)
    )


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )


    if len(
        np.unique(y_true)
    ) == 2:

        roc_auc = roc_auc_score(
            y_true,
            probability
        )

    else:

        roc_auc = np.nan


    results.append(
        {
            "subject": subject,
            "total_epochs": total_epochs,
            "normal_epochs": normal_epochs,
            "seizure_epochs": seizure_epochs,
            "seizure_prevalence":
                seizure_epochs / total_epochs,
            "accuracy": accuracy,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "roc_auc": roc_auc,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    )


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "subject"
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nSaved:",
    OUTPUT_FILE
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================================")
print("SUBJECT SUMMARY")
print("============================================================")

print(
    "Subjects analyzed:",
    len(results_df)
)


# ============================================================
# SENSITIVITY
# ============================================================

sensitivity = results_df[
    "sensitivity"
]

print(
    "\nMean sensitivity:",
    f"{sensitivity.mean():.4f}"
)

print(
    "Median sensitivity:",
    f"{sensitivity.median():.4f}"
)

print(
    "Subjects with sensitivity > 50%:",
    int(
        np.sum(
            sensitivity > 0.50
        )
    )
)

print(
    "Subjects with sensitivity > 80%:",
    int(
        np.sum(
            sensitivity > 0.80
        )
    )
)

print(
    "Subjects with sensitivity = 0:",
    int(
        np.sum(
            sensitivity == 0
        )
    )
)


# ============================================================
# SPECIFICITY
# ============================================================

specificity = results_df[
    "specificity"
]

print(
    "\nMean specificity:",
    f"{specificity.mean():.4f}"
)

print(
    "Median specificity:",
    f"{specificity.median():.4f}"
)

print(
    "Subjects with specificity > 90%:",
    int(
        np.sum(
            specificity > 0.90
        )
    )
)


# ============================================================
# ROC-AUC
# ============================================================

roc_values = results_df[
    "roc_auc"
].dropna()

print(
    "\nValid ROC-AUC subjects:",
    len(roc_values)
)

print(
    "Mean ROC-AUC:",
    f"{roc_values.mean():.4f}"
)

print(
    "Median ROC-AUC:",
    f"{roc_values.median():.4f}"
)


# ============================================================
# F1
# ============================================================

f1 = results_df[
    "f1"
]

print(
    "\nMean F1:",
    f"{f1.mean():.4f}"
)

print(
    "Median F1:",
    f"{f1.median():.4f}"
)


# ============================================================
# BEST SUBJECTS
# ============================================================

print("\n============================================================")
print("TOP 10 SUBJECTS BY SENSITIVITY")
print("============================================================")

print(
    results_df[
        [
            "subject",
            "seizure_epochs",
            "sensitivity",
            "specificity",
            "f1",
            "roc_auc"
        ]
    ]
    .sort_values(
        "sensitivity",
        ascending=False
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# WORST SUBJECTS
# ============================================================

print("\n============================================================")
print("10 SUBJECTS WITH LOWEST SENSITIVITY")
print("============================================================")

print(
    results_df[
        [
            "subject",
            "seizure_epochs",
            "sensitivity",
            "specificity",
            "f1",
            "roc_auc"
        ]
    ]
    .sort_values(
        "sensitivity",
        ascending=True
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# SEIZURE PREVALENCE
# ============================================================

print("\n============================================================")
print("SEIZURE PREVALENCE")
print("============================================================")

print(
    "Mean prevalence:",
    f"{results_df['seizure_prevalence'].mean():.4f}"
)

print(
    "Median prevalence:",
    f"{results_df['seizure_prevalence'].median():.4f}"
)

print(
    "Subjects with zero seizure epochs:",
    int(
        np.sum(
            results_df["seizure_epochs"] == 0
        )
    )
)


print(
    "\n============================================================"
)

print(
    "SUBJECT-WISE ANALYSIS COMPLETE"
)

print(
    "============================================================"
)