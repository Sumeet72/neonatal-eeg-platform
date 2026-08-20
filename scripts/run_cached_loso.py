import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from src.data.cached_eeg_dataset import CachedEEGDataset
from src.models.cnn import EEGCNN
from src.training.losses import get_weighted_loss
from src.training.trainer import train_model
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.evaluation.report import generate_report


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

METADATA_FILE = "results/stft_cache/metadata.csv"

PREDICTION_DIR = "results/loso_predictions"

RESULTS_FILE = "results/loso_results.csv"

BATCH_SIZE = 32
EPOCHS = 5
PATIENCE = 2
LEARNING_RATE = 1e-4

VAL_SIZE = 0.20
RANDOM_STATE = 42

# Safety switch:
# Start with 1 fold to test the complete runner.
MAX_FOLDS = 1


# ============================================================
# SETUP
# ============================================================

set_seed(SEED)

os.makedirs(
    PREDICTION_DIR,
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = get_device()

print("\nSelected device:", device)


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(
    METADATA_FILE
)

subjects = sorted(
    df["subject_id"].unique(),
    key=lambda x: (
        x != "N001",
        str(x)
    )
)
print("\n============================================================")
print("CACHED FULL LOSO")
print("============================================================")

print(
    "Total epochs:",
    len(df)
)

print(
    "Total subjects:",
    len(subjects)
)

print(
    "Folds to run:",
    min(MAX_FOLDS, len(subjects))
)


# ============================================================
# RESUME SUPPORT
# ============================================================

completed_subjects = set()

if os.path.exists(RESULTS_FILE):

    existing_results = pd.read_csv(
        RESULTS_FILE
    )

    if "subject" in existing_results.columns:

        completed_subjects = set(
            existing_results["subject"]
            .astype(str)
        )

print(
    "Already completed:",
    len(completed_subjects)
)


# ============================================================
# RESULTS
# ============================================================

results = []

fold_count = 0


# ============================================================
# LOSO LOOP
# ============================================================

for fold_number, test_subject in enumerate(
    subjects,
    start=1
):

    if test_subject in completed_subjects:

        print(
            f"\nSkipping {test_subject} "
            f"(already completed)"
        )

        continue

    if fold_count >= MAX_FOLDS:

        break

    fold_count += 1

    print(
        "\n============================================================"
    )

    print(
        f"FOLD {fold_number}/{len(subjects)}"
    )

    print(
        "============================================================"
    )

    print(
        "Test subject:",
        test_subject
    )


    # ========================================================
    # RESET SEED FOR EVERY FOLD
    # ========================================================

    set_seed(
        SEED
    )


    # ========================================================
    # TEST / REMAINING DATA
    # ========================================================

    test_df = df[
        df["subject_id"] == test_subject
    ].copy()

    remaining_df = df[
        df["subject_id"] != test_subject
    ].copy()


    print(
        "Test epochs:",
        len(test_df)
    )


    # ========================================================
    # TRAIN / VALIDATION SUBJECT SPLIT
    # ========================================================

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, val_idx = next(
        splitter.split(
            remaining_df,
            remaining_df["label"],
            groups=remaining_df["subject_id"]
        )
    )


    train_df = (
        remaining_df
        .iloc[train_idx]
        .reset_index(drop=True)
    )

    val_df = (
        remaining_df
        .iloc[val_idx]
        .reset_index(drop=True)
    )


    # ========================================================
    # SUBJECT LEAKAGE CHECK
    # ========================================================

    train_subjects = set(
        train_df["subject_id"]
    )

    val_subjects = set(
        val_df["subject_id"]
    )

    test_subjects = set(
        test_df["subject_id"]
    )


    assert not (
        train_subjects & val_subjects
    )

    assert not (
        train_subjects & test_subjects
    )

    assert not (
        val_subjects & test_subjects
    )


    print(
        "\nSubject split:"
    )

    print(
        "Training subjects:",
        len(train_subjects)
    )

    print(
        "Validation subjects:",
        len(val_subjects)
    )

    print(
        "Test subjects:",
        len(test_subjects)
    )


    print(
        "\nEpoch split:"
    )

    print(
        "Training epochs:",
        len(train_df)
    )

    print(
        "Validation epochs:",
        len(val_df)
    )

    print(
        "Test epochs:",
        len(test_df)
    )


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = CachedEEGDataset(
        train_df
    )

    val_dataset = CachedEEGDataset(
        val_df
    )

    test_dataset = CachedEEGDataset(
        test_df
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = EEGCNN(
        num_channels=9,
        num_classes=2
    ).to(device)


    print(
        "\nModel device:",
        next(model.parameters()).device
    )


    # ========================================================
    # WEIGHTED LOSS
    # ========================================================

    criterion, class_weights = get_weighted_loss(
        train_df["label"].to_numpy(copy=True),
        device
    )



    print(
        "\nClass weights:",
        class_weights
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ========================================================
    # TRAINING
    # ========================================================

    print(
        "\n============================================================"
    )

    print(
        "TRAINING"
    )

    print(
        "============================================================"
    )


    start_time = time.time()


    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE
    )


    training_time = (
        time.time() - start_time
    )


    print(
        f"\nTraining time: "
        f"{training_time / 60:.2f} minutes"
    )


    # ========================================================
    # TEST
    # ========================================================

    print(
        "\n============================================================"
    )

    print(
        "TEST"
    )

    print(
        "============================================================"
    )


    model.eval()


    all_true = []
    all_pred = []
    all_probability = []

    prediction_subjects = []
    prediction_epochs = []


    with torch.no_grad():

        for batch in test_loader:

            x = batch[
                "spectrogram"
            ].to(
                device,
                non_blocking=True
            )

            y = batch[
                "label"
            ].to(
                device,
                non_blocking=True
            )


            outputs = model(
                x
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            predictions = outputs.argmax(
                dim=1
            )


            seizure_probability = (
                probabilities[:, 1]
            )


            all_true.extend(
                y.cpu().numpy()
            )

            all_pred.extend(
                predictions.cpu().numpy()
            )

            all_probability.extend(
                seizure_probability.cpu().numpy()
            )

            prediction_subjects.extend(
                batch["subject_id"]
            )

            prediction_epochs.extend(
                batch["epoch_id"].numpy()
            )


    # ========================================================
    # ARRAYS
    # ========================================================

    y_true = np.asarray(
        all_true
    )

    y_pred = np.asarray(
        all_pred
    )

    y_probability = np.asarray(
        all_probability
    )


    # ========================================================
    # EVALUATION
    # ========================================================

    report = generate_report(
        y_true,
        y_pred,
        y_probability
    )


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
        "Accuracy:",
        f"{report['accuracy']:.4f}"
    )

    print(
        "Precision:",
        f"{report['precision']:.4f}"
    )

    print(
        "Sensitivity:",
        f"{report['sensitivity']:.4f}"
    )

    print(
        "Specificity:",
        f"{report['specificity']:.4f}"
    )

    print(
        "F1-score:",
        f"{report['f1']:.4f}"
    )

    print(
        "ROC-AUC:",
        (
            f"{report['roc_auc']:.4f}"
            if report["roc_auc"] is not None
            else "N/A"
        )
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame(
        {
            "subject_id": prediction_subjects,
            "epoch_id": prediction_epochs,
            "true_label": y_true,
            "predicted_label": y_pred,
            "seizure_probability": y_probability
        }
    )


    prediction_file = (
        f"{PREDICTION_DIR}/"
        f"{test_subject}.csv"
    )


    prediction_df.to_csv(
        prediction_file,
        index=False
    )


    print(
        "\nPredictions saved:",
        prediction_file
    )


    # ========================================================
    # STORE FOLD RESULT
    # ========================================================

    fold_result = {

        "subject": test_subject,

        "accuracy":
            report["accuracy"],

        "precision":
            report["precision"],

        "sensitivity":
            report["sensitivity"],

        "specificity":
            report["specificity"],

        "f1":
            report["f1"],

        "roc_auc":
            report["roc_auc"],

        "training_time_minutes":
            training_time / 60
    }


    results.append(
        fold_result
    )


    # ========================================================
    # SAVE RESULTS AFTER EVERY FOLD
    # ========================================================

    result_df = pd.DataFrame(
        results
    )


    if os.path.exists(RESULTS_FILE):

        existing_results = pd.read_csv(
            RESULTS_FILE
        )

        result_df = pd.concat(
            [
                existing_results,
                result_df
            ],
            ignore_index=True
        )

        result_df = (
            result_df
            .drop_duplicates(
                subset=["subject"],
                keep="last"
            )
        )


    result_df.to_csv(
        RESULTS_FILE,
        index=False
    )


    print(
        "\nFold complete:",
        test_subject
    )

    print(
        "Accuracy:",
        f"{report['accuracy']:.4f}"
    )

    print(
        "Sensitivity:",
        f"{report['sensitivity']:.4f}"
    )

    print(
        "Specificity:",
        f"{report['specificity']:.4f}"
    )

    print(
        "ROC-AUC:",
        (
            f"{report['roc_auc']:.4f}"
            if report["roc_auc"] is not None
            else "N/A"
        )
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n============================================================"
)

print(
    "CACHED LOSO RUN COMPLETE"
)

print(
    "============================================================"
)

print(
    "Folds executed:",
    fold_count
)

print(
    "Results:",
    RESULTS_FILE
)

print(
    "Predictions:",
    PREDICTION_DIR
)