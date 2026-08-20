import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.eeg_dataset import EEGDataset
from src.models.cnn import EEGCNN
from src.utils.device import get_device


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BATCH_SIZE = 4
LEARNING_RATE = 1e-4


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = get_device()


# --------------------------------------------------
# LOAD MASTER DATASET
# --------------------------------------------------

df = pd.read_csv(
    "results/master_epochs.csv"
)

# Only a few samples for the smoke test.
# We are NOT training the complete dataset yet.
test_df = df.iloc[:8].copy()

print(
    "\nSamples for smoke test:",
    len(test_df)
)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

dataset = EEGDataset(
    test_df,
    device="cpu"
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = EEGCNN(
    num_channels=9,
    num_classes=2
).to(device)


# --------------------------------------------------
# CLASS-WEIGHTED LOSS
# --------------------------------------------------

labels = df["label"].values

class_counts = (
    df["label"]
    .value_counts()
    .sort_index()
)

print(
    "\nClass counts:"
)

print(
    class_counts
)

weights = (
    len(labels)
    /
    (
        2 * class_counts
    )
)

class_weights = torch.tensor(
    weights.values,
    dtype=torch.float32,
    device=device
)

print(
    "\nClass weights:",
    class_weights
)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# TRAINING SMOKE TEST
# --------------------------------------------------

model.train()

for batch_number, batch in enumerate(loader):

    spectrogram = batch[
        "spectrogram"
    ].to(
        device,
        non_blocking=True
    )

    labels_batch = batch[
        "label"
    ].to(
        device,
        non_blocking=True
    )

    # ----------------------------------------------
    # Forward
    # ----------------------------------------------

    optimizer.zero_grad()

    outputs = model(
        spectrogram
    )

    loss = criterion(
        outputs,
        labels_batch
    )

    # ----------------------------------------------
    # Backward
    # ----------------------------------------------

    loss.backward()

    optimizer.step()

    predictions = (
        outputs.argmax(
            dim=1
        )
    )

    accuracy = (
        predictions
        == labels_batch
    ).float().mean()

    print(
        f"\nBatch {batch_number + 1}"
    )

    print(
        "Input:",
        spectrogram.shape
    )

    print(
        "Input device:",
        spectrogram.device
    )

    print(
        "Output:",
        outputs.shape
    )

    print(
        "Loss:",
        loss.item()
    )

    print(
        "Accuracy:",
        accuracy.item()
    )

    print(
        "GPU memory allocated:",
        round(
            torch.cuda.memory_allocated()
            / (1024 ** 2),
            2
        ),
        "MB"
    )

    # Only one actual training batch.
    break


print(
    "\nREAL EEG CNN TRAINING SMOKE TEST SUCCESSFUL."
)