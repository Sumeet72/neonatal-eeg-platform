import time

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from src.data.eeg_dataset import EEGDataset
from src.models.cnn import EEGCNN
from src.training.losses import get_weighted_loss
from src.training.trainer import train_model
from src.utils.device import get_device


# ============================================================
# SETTINGS
# ============================================================

TEST_SUBJECT = "N001"

BATCH_SIZE = 8
EPOCHS = 5
PATIENCE = 2
LEARNING_RATE = 1e-4

VAL_SIZE = 0.2
RANDOM_STATE = 42


# ============================================================
# DEVICE
# ============================================================

device = get_device()

print("\nSelected device:", device)


# ============================================================
# LOAD MASTER DATASET
# ============================================================

df = pd.read_csv(
    "results/master_epochs.csv"
)

print("\nMaster dataset:")
print("Epochs:", len(df))
print(
    "Subjects:",
    df["subject_id"].nunique()
)


# ============================================================
# OUTER LOSO TEST SUBJECT
# ============================================================

test_df = df[
    df["subject_id"] == TEST_SUBJECT
].copy()

remaining_df = df[
    df["subject_id"] != TEST_SUBJECT
].copy()

print("\n============================================================")
print("LOSO FOLD")
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
# TRAIN / VALIDATION SPLIT BY SUBJECT
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

train_df = remaining_df.iloc[
    train_idx
].reset_index(drop=True)

val_df = remaining_df.iloc[
    val_idx
].reset_index(drop=True)


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

print("\nCreating datasets...")

train_dataset = EEGDataset(
    train_df,
    device="cpu"
)

val_dataset = EEGDataset(
    val_df,
    device="cpu"
)

test_dataset = EEGDataset(
    test_df,
    device="cpu"
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
    next(
        model.parameters()
    ).device
)


# ============================================================
# CLASS-WEIGHTED LOSS
# IMPORTANT: TRAINING LABELS ONLY
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
# TRAIN
# ============================================================

print("\n============================================================")
print("TRAINING")
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

training_time = time.time() - start_time


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

correct = 0
total = 0

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

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)


test_accuracy = (
    correct / total
    if total > 0
    else 0
)


print(
    "Test epochs:",
    total
)

print(
    "Test accuracy:",
    f"{test_accuracy:.4f}"
)


# ============================================================
# FINAL
# ============================================================

print("\n============================================================")
print("ONE LOSO FOLD COMPLETE")
print("============================================================")

print(
    "Test subject:",
    TEST_SUBJECT
)

print(
    "Training time:",
    f"{training_time / 60:.2f} minutes"
)

print(
    "Test accuracy:",
    f"{test_accuracy:.4f}"
)

print(
    "\nONE-FOLD LOSO TEST SUCCESSFUL."
)