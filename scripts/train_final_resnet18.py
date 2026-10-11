import os
import time
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.cached_eeg_dataset import CachedEEGDataset
from src.models.ResNet18 import ResNet18EEG
from src.training.losses import get_weighted_loss
from src.utils.device import get_device
from src.utils.seed import set_seed


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

METADATA_FILE = "results/stft_cache/metadata.csv"

OUTPUT_DIR = "results/final_model"
CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    "ResNet18_final.pt"
)

METADATA_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "model_metadata.json"
)

BATCH_SIZE = 32

# Same maximum training configuration used during LOSO
EPOCHS = 5

LEARNING_RATE = 1e-4

THRESHOLD = 0.40


# ============================================================
# SETUP
# ============================================================

set_seed(SEED)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

device = get_device()

print("=" * 70)
print("FINAL RESNET-18 TRAINING")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD FULL COMBINED DATASET
# ============================================================

print("\nLoading metadata...")

df = pd.read_csv(
    METADATA_FILE
)

print("\nDataset summary")
print("-" * 70)

print("Total epochs:", len(df))
print(
    "Total subjects:",
    df["subject_id"].nunique()
)

print(
    "Normal epochs:",
    int((df["label"] == 0).sum())
)

print(
    "Seizure epochs:",
    int((df["label"] == 1).sum())
)


# ============================================================
# DATASET
# ============================================================

dataset = CachedEEGDataset(
    df
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

print("\nCreating ResNet-18...")

model = ResNet18EEG(
    num_channels=9,
    num_classes=2
).to(device)

print(
    "Model device:",
    next(model.parameters()).device
)


# ============================================================
# WEIGHTED LOSS
# ============================================================

criterion, class_weights = get_weighted_loss(
    df["label"].to_numpy(copy=True),
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

print("\n" + "=" * 70)
print("TRAINING FINAL DEPLOYMENT MODEL")
print("=" * 70)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

start_time = time.time()

history = {
    "epoch": [],
    "loss": [],
    "accuracy": []
}


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

        x = batch["spectrogram"].to(
            device,
            non_blocking=True
        )

        y = batch["label"].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * x.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    history["epoch"].append(
        epoch
    )

    history["loss"].append(
        epoch_loss
    )

    history["accuracy"].append(
        epoch_accuracy
    )

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {epoch_accuracy:.4f}"
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

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "model_name":
        "ResNet18EEG",

    "num_channels":
        9,

    "num_classes":
        2,

    "input_shape":
        [9, 129, 59],

    "threshold":
        THRESHOLD,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "epochs":
        EPOCHS,

    "seed":
        SEED,

    "class_weights":
        class_weights.detach().cpu(),

    "training_samples":
        len(df),

    "training_subjects":
        int(df["subject_id"].nunique())
}


torch.save(
    checkpoint,
    CHECKPOINT_FILE
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_file = os.path.join(
    OUTPUT_DIR,
    "ResNet18_final_training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False
)


# ============================================================
# SAVE MODEL METADATA
# ============================================================

metadata = {

    "model":
        "ResNet-18",

    "dataset":
        "Combined Nathan + Sumit",

    "subjects":
        int(df["subject_id"].nunique()),

    "epochs":
        int(len(df)),

    "normal_epochs":
        int((df["label"] == 0).sum()),

    "seizure_epochs":
        int((df["label"] == 1).sum()),

    "input_shape":
        [9, 129, 59],

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "training_epochs":
        EPOCHS,

    "operating_threshold":
        THRESHOLD,

    "training_time_minutes":
        training_time / 60
}


with open(
    METADATA_OUTPUT,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    "\nCheckpoint:"
)

print(
    CHECKPOINT_FILE
)

print(
    "\nTraining history:"
)

print(
    history_file
)

print(
    "\nModel metadata:"
)

print(
    METADATA_OUTPUT
)

print(
    "\nOperating threshold:",
    THRESHOLD
)

print("=" * 70)