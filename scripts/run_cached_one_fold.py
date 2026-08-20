import time

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

SEED = 42

set_seed(SEED)
import numpy as np

from src.evaluation.report import (
    generate_report,
    print_report
)

# ============================================================
# SETTINGS
# ============================================================

METADATA_FILE = (
    "results/stft_cache/metadata.csv"
)

TEST_SUBJECT = "N001"

BATCH_SIZE = 32
EPOCHS = 5
PATIENCE = 2
LEARNING_RATE = 1e-4

VAL_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# DEVICE
# ============================================================

device = get_device()

print("\nSelected device:", device)


# ============================================================
# LOAD CACHED METADATA
# ============================================================

df = pd.read_csv(
    METADATA_FILE
)

print("\nCached dataset:")
print("Epochs:", len(df))
print(
    "Subjects:",
    df["subject_id"].nunique()
)


# ============================================================
# TEST SUBJECT
# ============================================================

test_df = df[
    df["subject_id"] == TEST_SUBJECT
].copy()

remaining_df = df[
    df["subject_id"] != TEST_SUBJECT
].copy()


print("\n============================================================")
print("CACHED LOSO FOLD")
print("============================================================")

print(
    "Test subject:",
    TEST_SUBJECT
)

print(
    "Test epochs:",
    len(test_df)
)


# ============================================================
# TRAIN / VALIDATION SUBJECT SPLIT
# ============================================================

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


# ============================================================
# CHECK SUBJECT LEAKAGE
# ============================================================

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


print("\nSubject split:")
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


print("\nEpoch split:")
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


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nTraining labels:")
print(
    train_df["label"].value_counts()
)

print("\nValidation labels:")
print(
    val_df["label"].value_counts()
)

print("\nTest labels:")
print(
    test_df["label"].value_counts()
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = CachedEEGDataset(
    train_df
)

val_dataset = CachedEEGDataset(
    val_df
)

test_dataset = CachedEEGDataset(
    test_df
)


# ============================================================
# DATALOADERS
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

model = EEGCNN(
    num_channels=9,
    num_classes=2
).to(device)

print(
    "\nModel device:",
    next(model.parameters()).device
)


# ============================================================
# CLASS-WEIGHTED LOSS
# TRAINING DATA ONLY
# ============================================================

criterion, class_weights = get_weighted_loss(
    train_df["label"].values,
    device
)

print(
    "\nClass weights:",
    class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n============================================================")
print("CACHED TRAINING")
print("============================================================")

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


# ============================================================
# TEST
# ============================================================

print("\n============================================================")
print("TEST")
print("============================================================")


model.eval()

all_true = []
all_pred = []
all_probability = []

test_subjects = []
test_epochs = []


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

        outputs = model(x)

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

        test_subjects.extend(
            batch["subject_id"]
        )

        test_epochs.extend(
            batch["epoch_id"].numpy()
        )


# ============================================================
# NUMPY ARRAYS
# ============================================================

y_true = np.asarray(
    all_true
)

y_pred = np.asarray(
    all_pred
)

y_probability = np.asarray(
    all_probability
)


# ============================================================
# EVALUATION
# ============================================================

report = generate_report(
    y_true,
    y_pred,
    y_probability
)

print_report(
    report
)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "subject_id": test_subjects,
        "epoch_id": test_epochs,
        "true_label": y_true,
        "predicted_label": y_pred,
        "seizure_probability": y_probability
    }
)

prediction_dir = (
    "results/loso_predictions"
)

import os

os.makedirs(
    prediction_dir,
    exist_ok=True
)

prediction_file = (
    f"{prediction_dir}/"
    f"{TEST_SUBJECT}.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print(
    "\nPredictions saved:",
    prediction_file
)

# ============================================================
# TEST SUMMARY
# ============================================================

print(
    "\nTest samples:",
    len(y_true)
)

print(
    "Test subject:",
    TEST_SUBJECT
)

print(
    "Test accuracy:",
    f"{report['accuracy']:.4f}"
)

print(
    "Test sensitivity:",
    f"{report['sensitivity']:.4f}"
)

print(
    "Test specificity:",
    f"{report['specificity']:.4f}"
)

if report["roc_auc"] is not None:

    print(
        "Test ROC-AUC:",
        f"{report['roc_auc']:.4f}"
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n============================================================"
)

print(
    "CACHED ONE-FOLD LOSO COMPLETE"
)

print(
    "============================================================"
)

print(
    "Test subject:",
    TEST_SUBJECT
)

print(
    "Training time:",
    f"{training_time / 60:.2f} minutes"
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
    "Precision:",
    f"{report['precision']:.4f}"
)

print(
    "F1-score:",
    f"{report['f1']:.4f}"
)

if report["roc_auc"] is not None:

    print(
        "ROC-AUC:",
        f"{report['roc_auc']:.4f}"
    )