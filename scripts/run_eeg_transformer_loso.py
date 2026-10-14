"""Run a subject-level LOSO experiment for the pure EEG Transformer.

Default behavior runs one outer fold as a smoke/pilot run. Set
MAX_LOSO_FOLDS=132 to run/resume the full LOSO experiment.

This script uses the existing STFT cache. It does not rebuild or modify it.
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from src.data.cached_eeg_dataset import CachedEEGDataset
from src.models.eeg_transformer import EEGSpatioTemporalTransformer
from src.training.losses import get_weighted_loss
from src.training.trainer import train_model
from src.utils.device import get_device
from src.utils.seed import set_seed


SEED = 42
METADATA_FILE = Path("results/stft_cache/metadata.csv")
OUTPUT_DIR = Path("results/EEGTransformer_LOSO")
PREDICTION_DIR = OUTPUT_DIR / "predictions"
HISTORY_DIR = OUTPUT_DIR / "training_history"
FOLD_METRICS_FILE = OUTPUT_DIR / "fold_metrics.csv"
POOLED_METRICS_FILE = OUTPUT_DIR / "pooled_metrics.json"

BATCH_SIZE = int(os.environ.get("TRANSFORMER_BATCH_SIZE", "8"))
EPOCHS = int(os.environ.get("TRANSFORMER_EPOCHS", "5"))
PATIENCE = int(os.environ.get("TRANSFORMER_PATIENCE", "2"))
LEARNING_RATE = float(os.environ.get("TRANSFORMER_LEARNING_RATE", "0.0001"))
MAX_LOSO_FOLDS = int(os.environ.get("MAX_LOSO_FOLDS", "1"))
VALIDATION_SIZE = 0.20
THRESHOLD_GRID = np.round(np.arange(0.10, 0.901, 0.05), 2)


def safe_filename(value: str) -> str:
    """Return a filesystem-safe identifier for an output filename."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "subject"


def calculate_metrics(y_true, y_pred, probabilities):
    """Calculate binary classification metrics, safely handling one-class data."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    has_both_classes = len(np.unique(y_true)) == 2
    roc_auc = (
        float(roc_auc_score(y_true, probabilities))
        if has_both_classes else None
    )
    pr_auc = (
        float(average_precision_score(y_true, probabilities))
        if np.any(y_true == 1) else None
    )

    return {
        "n_epochs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.inference_mode()
def collect_predictions(model, loader, device):
    """Get epoch labels and positive-class probabilities without gradient tracking."""
    model.eval()
    labels, probabilities, subjects, epochs, datasets = [], [], [], [], []

    for batch in loader:
        x = batch["spectrogram"].to(device, non_blocking=True)
        y = batch["label"].cpu().numpy().astype(int)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        labels.extend(y.tolist())
        probabilities.extend(probs.astype(float).tolist())
        subjects.extend([str(value) for value in batch["subject_id"]])
        epochs.extend(batch["epoch_id"].cpu().numpy().astype(int).tolist())
        datasets.extend([str(value) for value in batch["dataset"]])

    return {
        "true_label": np.asarray(labels, dtype=int),
        "seizure_probability": np.asarray(probabilities, dtype=float),
        "subject_id": subjects,
        "epoch_id": epochs,
        "dataset": datasets,
    }


def select_threshold_from_validation(y_true, probabilities):
    """Select a threshold on validation subjects only, maximizing validation F1."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    # If validation data has no positive examples, it cannot choose a useful
    # seizure threshold. Use the predeclared fallback and record the limitation.
    if not np.any(y_true == 1):
        return 0.40, None, "fallback_0.40_validation_has_no_positive_epochs"

    best_threshold = 0.40
    best_f1 = -1.0

    for threshold in THRESHOLD_GRID:
        prediction = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, prediction, zero_division=0)

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1, "validation_f1_grid_search"


def write_pooled_metrics(metadata, device):
    """Aggregate only completed outer-fold prediction files."""
    if not FOLD_METRICS_FILE.exists():
        print("No completed folds yet; pooled metrics were not generated.")
        return

    folds = pd.read_csv(FOLD_METRICS_FILE)
    if folds.empty:
        print("No completed folds yet; pooled metrics were not generated.")
        return

    frames = []
    for subject in folds["subject_id"].astype(str):
        prediction_path = PREDICTION_DIR / f"{safe_filename(subject)}.csv"
        if prediction_path.exists():
            frames.append(pd.read_csv(prediction_path))

    if not frames:
        print("No prediction files found for completed fold summaries.")
        return

    combined = pd.concat(frames, ignore_index=True)
    metrics = calculate_metrics(
        combined["true_label"].to_numpy(),
        combined["predicted_label"].to_numpy(),
        combined["seizure_probability"].to_numpy(),
    )

    total_subjects = int(metadata["subject_id"].nunique())
    evaluated_subjects = int(combined["subject_id"].nunique())
    thresholds = folds["validation_threshold"].dropna().astype(float)

    summary = {
        "model": "Pure factorized spatio-temporal EEG Transformer",
        "evaluation_status": (
            "complete_loso"
            if evaluated_subjects == total_subjects
            else "partial_loso_pilot"
        ),
        "evaluated_subjects": evaluated_subjects,
        "total_subjects": total_subjects,
        "evaluated_epochs": int(len(combined)),
        "dataset_epoch_count": int(len(metadata)),
        "normal_epochs_in_evaluated_predictions": int((combined["true_label"] == 0).sum()),
        "seizure_epochs_in_evaluated_predictions": int((combined["true_label"] == 1).sum()),
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "threshold_method": "per-fold threshold selected using validation subjects only by F1 grid search; fallback 0.40 if validation has no positives",
        "median_validation_threshold": float(thresholds.median()) if not thresholds.empty else None,
        "thresholds_min": float(thresholds.min()) if not thresholds.empty else None,
        "thresholds_max": float(thresholds.max()) if not thresholds.empty else None,
        "sumit_label_caveat": (
            "Before interpreting metrics, verify that Sumit recording-level seizures_YN labels "
            "are valid for the specific 30-second cached segment. Cache/master consistency alone "
            "does not establish segment-level ground-truth validity."
        ),
        **metrics,
    }

    POOLED_METRICS_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("TRANSFORMER LOSO SUMMARY")
    print("=" * 72)
    print(f"Evaluation status: {summary['evaluation_status']}")
    print(f"Subjects evaluated: {evaluated_subjects}/{total_subjects}")
    print(f"Epoch predictions: {len(combined):,}")
    for key in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "pr_auc"):
        value = summary.get(key)
        print(f"{key}: {'N/A' if value is None else f'{value:.4f}'}")
    print("Saved:", POOLED_METRICS_FILE)
    if evaluated_subjects != total_subjects:
        print("NOTE: This is a partial/pilot result, not a completed LOSO evaluation.")


def main():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Required cached STFT metadata is missing: {METADATA_FILE}. "
            "Use the existing validated cache; do not rebuild it unless required."
        )

    if MAX_LOSO_FOLDS < 1:
        raise ValueError("MAX_LOSO_FOLDS must be at least 1.")

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = get_device()

    metadata = pd.read_csv(METADATA_FILE)
    required_columns = {
        "cache_file", "epoch_id", "subject_id", "dataset", "label"
    }
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"Cache metadata is missing required columns: {sorted(missing)}")
    if metadata[list(required_columns)].isna().any().any():
        raise ValueError("Cache metadata contains missing required values.")
    if not set(metadata["label"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0=normal and 1=seizure.")

    subjects = sorted(metadata["subject_id"].astype(str).unique().tolist())
    print("=" * 72)
    print("PURE EEG SPATIO-TEMPORAL TRANSFORMER — SUBJECT-LEVEL LOSO")
    print("=" * 72)
    print("Device:", device)
    print("Subjects:", len(subjects))
    print("Epochs:", len(metadata))
    print("Labels:", metadata["label"].value_counts().sort_index().to_dict())
    print("Batch size:", BATCH_SIZE, "| Epochs/fold:", EPOCHS)
    print("Outer folds allowed in this invocation:", MAX_LOSO_FOLDS)
    print("Using cached STFT tensors; no cache rebuild.")
    print(
        "SCIENTIFIC CAVEAT: verify Sumit recording-level label alignment "
        "to its selected 30-second cached segments before interpreting model metrics."
    )

    if FOLD_METRICS_FILE.exists():
        prior = pd.read_csv(FOLD_METRICS_FILE)
        completed_subjects = set(prior["subject_id"].astype(str))
    else:
        prior = pd.DataFrame()
        completed_subjects = set()

    completed_this_invocation = 0

    for fold_number, test_subject in enumerate(subjects, start=1):
        if fold_number > MAX_LOSO_FOLDS:
            break
        if test_subject in completed_subjects:
            print(f"[{fold_number}/{len(subjects)}] Already complete; skipping {test_subject}")
            continue

        started = time.time()
        print("\n" + "-" * 72)
        print(f"FOLD {fold_number}/{len(subjects)} — held-out subject: {test_subject}")
        print("-" * 72)

        set_seed(SEED + fold_number)

        test_df = metadata[
            metadata["subject_id"].astype(str) == test_subject
        ].copy().reset_index(drop=True)
        remaining_df = metadata[
            metadata["subject_id"].astype(str) != test_subject
        ].copy().reset_index(drop=True)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_SIZE,
            random_state=SEED,
        )
        train_idx, validation_idx = next(
            splitter.split(
                remaining_df,
                remaining_df["label"].astype(int),
                groups=remaining_df["subject_id"].astype(str),
            )
        )
        train_df = remaining_df.iloc[train_idx].copy().reset_index(drop=True)
        validation_df = remaining_df.iloc[validation_idx].copy().reset_index(drop=True)

        train_subjects = set(train_df["subject_id"].astype(str))
        validation_subjects = set(validation_df["subject_id"].astype(str))
        test_subjects = set(test_df["subject_id"].astype(str))
        if train_subjects & validation_subjects or train_subjects & test_subjects or validation_subjects & test_subjects:
            raise RuntimeError("Subject leakage detected in train/validation/test split.")
        if train_df["label"].nunique() < 2:
            raise RuntimeError(
                f"Training split for held-out subject {test_subject} has only one class."
            )

        print(
            f"Epochs — train: {len(train_df)}, validation: {len(validation_df)}, test: {len(test_df)}"
        )
        print(
            f"Subjects — train: {len(train_subjects)}, validation: {len(validation_subjects)}, test: 1"
        )

        train_loader = DataLoader(
            CachedEEGDataset(train_df),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        validation_loader = DataLoader(
            CachedEEGDataset(validation_df),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        test_loader = DataLoader(
            CachedEEGDataset(test_df),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        model = EEGSpatioTemporalTransformer(
            num_channels=9,
            num_frequency_bins=129,
            num_time_frames=59,
            num_classes=2,
            embed_dim=64,
            num_heads=4,
            temporal_layers=2,
            channel_layers=1,
            feedforward_dim=256,
            dropout=0.2,
        ).to(device)

        criterion, class_weights = get_weighted_loss(
            train_df["label"].to_numpy(copy=True),
            device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4,
        )

        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=EPOCHS,
            patience=PATIENCE,
        )

        subject_file = safe_filename(test_subject)
        pd.DataFrame(history).to_csv(
            HISTORY_DIR / f"{subject_file}.csv",
            index=False,
        )

        validation_output = collect_predictions(model, validation_loader, device)
        threshold, validation_f1, threshold_method = select_threshold_from_validation(
            validation_output["true_label"],
            validation_output["seizure_probability"],
        )
        print(
            f"Validation-selected threshold: {threshold:.2f} "
            f"(validation F1: {'N/A' if validation_f1 is None else f'{validation_f1:.4f}'})"
        )
        print("Threshold method:", threshold_method)

        test_output = collect_predictions(model, test_loader, device)
        probabilities = test_output["seizure_probability"]
        predicted = (probabilities >= threshold).astype(int)
        test_metrics = calculate_metrics(
            test_output["true_label"],
            predicted,
            probabilities,
        )

        prediction_df = pd.DataFrame({
            "subject_id": test_output["subject_id"],
            "dataset": test_output["dataset"],
            "epoch_id": test_output["epoch_id"],
            "true_label": test_output["true_label"],
            "seizure_probability": probabilities,
            "threshold_used": threshold,
            "predicted_label": predicted,
        })
        prediction_df.to_csv(
            PREDICTION_DIR / f"{subject_file}.csv",
            index=False,
        )

        fold_row = {
            "fold_number": fold_number,
            "subject_id": test_subject,
            "train_subjects": len(train_subjects),
            "validation_subjects": len(validation_subjects),
            "train_epochs": len(train_df),
            "validation_epochs": len(validation_df),
            "test_epochs": len(test_df),
            "validation_threshold": threshold,
            "validation_f1_at_threshold": validation_f1,
            "threshold_method": threshold_method,
            "training_seconds": time.time() - started,
            "class_weight_normal": float(class_weights[0].detach().cpu()),
            "class_weight_seizure": float(class_weights[1].detach().cpu()),
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }

        existing = (
            pd.read_csv(FOLD_METRICS_FILE)
            if FOLD_METRICS_FILE.exists()
            else pd.DataFrame()
        )
        updated = pd.concat(
            [existing, pd.DataFrame([fold_row])],
            ignore_index=True,
        ).drop_duplicates(subset=["subject_id"], keep="last")
        updated.to_csv(FOLD_METRICS_FILE, index=False)
        completed_subjects.add(test_subject)
        completed_this_invocation += 1

        print(
            f"Test accuracy={test_metrics['accuracy']:.4f}, "
            f"sensitivity={test_metrics['sensitivity']:.4f}, "
            f"F1={test_metrics['f1']:.4f}"
        )
        print("Saved fold predictions and metrics.")

        # Persist pooled summaries after every fold so interrupted runs can resume.
        write_pooled_metrics(metadata, device)

    print("\nRun finished.")
    print("New folds completed this invocation:", completed_this_invocation)
    print("Fold metrics:", FOLD_METRICS_FILE)
    print("Pooled summary:", POOLED_METRICS_FILE)
    print(
        "To continue/resume the full 132-subject experiment, set "
        "MAX_LOSO_FOLDS=132 and rerun this module."
    )


if __name__ == "__main__":
    main()
