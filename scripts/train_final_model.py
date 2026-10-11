import os
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


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

METADATA_FILE = (
    "results/stft_cache/metadata.csv"
)

BATCH_SIZE = 32
EPOCHS = 5
PATIENCE = 2
LEARNING_RATE = 1e-4

VAL_SIZE = 0.20
RANDOM_STATE = 42

CHECKPOINT_DIR = (
    "results/checkpoints"
)

CHECKPOINT_FILE = (
    f"{CHECKPOINT_DIR}/eegcnn_final.pth"
)


# ============================================================
# SEED
# ============================================================

set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = get_device()

print(
    "\nSelected device:",
    device
)


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(
    METADATA_FILE
)

print(
    "\n============================================================"
)

print(
    "FINAL CNN TRAINING"
)

print(
    "============================================================"
)

print(
    "Total epochs:",
    len(df)
)

print(
    "Total subjects:",
    df["subject_id"].nunique()
)


# ============================================================
# SUBJECT-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    splitter.split(
        df,
        df["label"],
        groups=df["subject_id"]
    )
)


train_df = (
    df
    .iloc[train_idx]
    .reset_index(drop=True)
)

val_df = (
    df
    .iloc[val_idx]
    .reset_index(drop=True)
)


print(
    "\nSubject split:"
)

print(
    "Training subjects:",
    train_df["subject_id"].nunique()
)

print(
    "Validation subjects:",
    val_df["subject_id"].nunique()
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


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print(
    "\nTraining labels:"
)

print(
    train_df["label"].value_counts()
    .sort_index()
)

print(
    "\nValidation labels:"
)

print(
    val_df["label"].value_counts()
    .sort_index()
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
# WEIGHTED LOSS
# ============================================================

criterion, class_weights = get_weighted_loss(
    train_df["label"].to_numpy(copy=True),
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

print(
    "\n============================================================"
)

print(
    "TRAINING FINAL CNN"
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


# ============================================================
# SAVE CHECKPOINT
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_name": "EEGCNN",
    "num_channels": 9,
    "num_classes": 2,
    "input_shape": [
        9,
        129,
        59
    ],
    "threshold": 0.40,
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "learning_rate": LEARNING_RATE,
    "class_weights": class_weights.cpu(),
    "training_subjects": train_df[
        "subject_id"
    ].nunique(),
    "validation_subjects": val_df[
        "subject_id"
    ].nunique(),
    "training_epochs": len(train_df),
    "validation_epochs": len(val_df),
    "history": history
}


torch.save(
    checkpoint,
    CHECKPOINT_FILE
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n============================================================"
)

print(
    "FINAL CNN CHECKPOINT SAVED"
)

print(
    "============================================================"
)

print(
    "Checkpoint:",
    CHECKPOINT_FILE
)

print(
    "Size:",
    f"{os.path.getsize(CHECKPOINT_FILE) / (1024 * 1024):.2f} MB"
)

print(
    "\nFinal CNN training complete."
)