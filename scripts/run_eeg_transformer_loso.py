"""Unified matched-split LOSO runner for the all-recording Nathan + Sumit experiment.

Select model with EEG_MIL_MODEL=transformer or resnet18. Reads the existing corrected
STFT cache and corrected manifest, includes all 169 Sumit recordings in training,
uses window labels where known, and applies max-pooling MIL to the three Sumit
recordings marked seizure-positive. Held-out test subjects are Nathan only.

This replaces the prior Transformer-only runner; it does not create another script
inside the project. Each model's outputs are stored in a separate results folder.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
        balanced_accuracy_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from src.data.cached_eeg_dataset import CachedEEGDataset
from src.models.ResNet18 import ResNet18EEG
from src.models.eeg_transformer import EEGSpatioTemporalTransformer
from src.utils.device import get_device
from src.utils.seed import set_seed

SEED = 42
MODEL_NAME = os.environ.get("EEG_MIL_MODEL", "transformer").strip().lower()
if MODEL_NAME not in {"transformer", "resnet18"}:
    raise ValueError("EEG_MIL_MODEL must be 'transformer' or 'resnet18'.")

METADATA_FILE = Path(os.environ.get(
    "EEG_MIL_METADATA_FILE", "results/corrected_epoch_dataset/stft_cache/metadata.csv"
))
MANIFEST_FILE = Path(os.environ.get(
    "EEG_MIL_MANIFEST_FILE", "results/corrected_epoch_dataset/master_epochs.csv"
))
SUMIT_TABLE_FILE = Path("results/sumit_epochs.csv")

INCLUDE_SUMIT_TRAINING = os.environ.get(
    "EEG_INCLUDE_SUMIT_TRAINING", "true"
).strip().lower() in {"1", "true", "yes", "on"}
OUTPUT_DIR = Path(os.environ.get(
    "EEG_MIL_OUTPUT_DIR", f"results/AllSumitMIL_{MODEL_NAME}_Nathan_LOSO"
))
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"
HISTORY_FILE = OUTPUT_DIR / "training_history.csv"
FOLD_METRICS_FILE = OUTPUT_DIR / "fold_metrics.csv"
POOLED_METRICS_FILE = OUTPUT_DIR / "pooled_metrics.json"

BATCH_SIZE = int(os.environ.get("EEG_MIL_BATCH_SIZE", "8"))
EPOCHS = int(os.environ.get("EEG_MIL_EPOCHS", "5"))
PATIENCE = int(os.environ.get("EEG_MIL_PATIENCE", "2"))
LEARNING_RATE = float(os.environ.get("EEG_MIL_LEARNING_RATE", "0.0001"))
MAX_LOSO_FOLDS = int(os.environ.get("MAX_LOSO_FOLDS", "1"))
VALIDATION_SIZE = 0.20
BAG_REPEATS = int(os.environ.get("EEG_MIL_BAG_REPEATS", "5"))
BAG_LOSS_WEIGHT = float(os.environ.get("EEG_MIL_BAG_LOSS_WEIGHT", "1.0"))
RESNET_PRETRAINED = os.environ.get("EEG_MIL_RESNET_PRETRAINED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
THRESHOLD_GRID = np.array([
    0.001, 0.002, 0.005, 0.0075,
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.10, 0.15,
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90
])

def calculate_metrics(y_true, y_pred, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(y_true):
        return {
            "n_epochs": 0, "accuracy": None, "precision": None,
            "sensitivity": None, "specificity": None, "f1": None,
            "roc_auc": None, "pr_auc": None, "balanced_accuracy": None,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n_epochs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)) if (tp + fn) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, probabilities)) if both_classes else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both_classes else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_threshold(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) == 0 or not np.any(y_true == 1):
        return 0.40, None, "fallback_0.40_validation_has_no_positive_windows"
    best_threshold, best_f1 = 0.40, -1.0
    for threshold in THRESHOLD_GRID:
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1, "validation_f1_grid_search"

def select_sumit_recording_threshold(y_true, probabilities):
    """
    Select threshold using balanced accuracy on Nathan validation recordings.
    Avoid thresholds that predict only one class.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return 0.50, None, "fallback_0.50_validation_has_one_class"

    best_threshold = None
    best_balanced_accuracy = -1.0
    best_f1 = -1.0

    for threshold in THRESHOLD_GRID:
        predictions = (probabilities >= threshold).astype(int)

        if np.unique(predictions).size < 2:
            continue

        balanced_acc = balanced_accuracy_score(
            y_true, predictions
        )
        current_f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        if (
            balanced_acc > best_balanced_accuracy
            or (
                np.isclose(balanced_acc, best_balanced_accuracy)
                and current_f1 > best_f1
            )
        ):
            best_threshold = float(threshold)
            best_balanced_accuracy = float(balanced_acc)
            best_f1 = float(current_f1)

    if best_threshold is None:
        return 0.50, None, "fallback_0.50_no_mixed_validation_predictions"

    return (
        best_threshold,
        best_f1,
        "validation_balanced_accuracy_grid_search",
    )

def make_model(device):
    if MODEL_NAME == "resnet18":
        return ResNet18EEG(
            num_channels=9, num_classes=2,
            pretrained=RESNET_PRETRAINED, dropout=0.4,
        ).to(device)
    return EEGSpatioTemporalTransformer().to(device)


def class_weighted_criterion(labels, device):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2).astype(float)
    if np.any(counts == 0):
        raise RuntimeError(f"Strong-label training data must include both classes; counts={counts.tolist()}")
    weights = counts.sum() / (2.0 * counts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor), weights.tolist()


@torch.inference_mode()
def collect_predictions(model, rows, device, include_unknown=False):
    if include_unknown:
        rows = rows.copy().reset_index(drop=True)
    else:
        rows = rows.loc[
            rows["label"].astype(int).isin([0, 1])
        ].copy().reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame(columns=["subject_id", "dataset", "epoch_id", "true_label", "seizure_probability"])
    loader = DataLoader(
        CachedEEGDataset(rows), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    model.eval()
    output = []
    for batch in loader:
        x = batch["spectrogram"].to(device, non_blocking=True)
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels = batch["label"].cpu().numpy().astype(int)
        for idx, label in enumerate(labels):
            output.append({
                "subject_id": str(batch["subject_id"][idx]),
                "dataset": str(batch["dataset"][idx]),
                "epoch_id": int(batch["epoch_id"][idx]),
                "true_label": int(label),
                "seizure_probability": float(probabilities[idx]),
            })
    return pd.DataFrame(output)


def run_positive_bag_update(model, optimizer, bag_rows, device):
    """Max-pooling MIL: a recording is positive if at least one window is positive."""
    bag_rows = bag_rows.sort_values(["start_second", "epoch_id"]).reset_index(drop=True)
    if bag_rows.empty:
        return None
    model.eval()  # Disable dropout/BatchNorm updates during the differentiable bag pass.
    optimizer.zero_grad(set_to_none=True)
    margin_chunks = []
    cache_paths = bag_rows["cache_file"].astype(str).tolist()
    for start in range(0, len(cache_paths), BATCH_SIZE):
        tensors = [
            torch.load(path, map_location="cpu", weights_only=True).float()
            for path in cache_paths[start:start + BATCH_SIZE]
        ]
        x = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        logits = checkpoint(model, x, use_reentrant=False)
        margin_chunks.append(logits[:, 1] - logits[:, 0])
        del x, tensors
    bag_logit = torch.cat(margin_chunks, dim=0).max().reshape(1)
    target = torch.ones(1, dtype=torch.float32, device=device)
    loss = nn.functional.binary_cross_entropy_with_logits(bag_logit, target) * BAG_LOSS_WEIGHT
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    model.train()
    return float(loss.detach().cpu())

def validation_loss(model, rows, device):
    rows = rows.loc[
        rows["label"].astype(int).isin([0, 1])
    ].copy().reset_index(drop=True)

    if rows.empty:
        return math.inf, None, None

    loader = DataLoader(
        CachedEEGDataset(rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss()
    model.eval()

    loss_sum, seen, correct = 0.0, 0, 0
    all_labels = []
    all_probabilities = []

    with torch.inference_mode():
        for batch in loader:
            x = batch["spectrogram"].to(
                device, non_blocking=True
            )
            y = batch["label"].to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            probabilities = torch.softmax(
                logits, dim=1
            )[:, 1]

            loss_sum += float(loss.item()) * len(y)
            seen += len(y)
            correct += int(
                (logits.argmax(1) == y).sum().item()
            )

            all_labels.extend(
                y.cpu().numpy().astype(int).tolist()
            )
            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    val_loss = loss_sum / seen
    val_accuracy = correct / seen

    val_ap = (
        float(average_precision_score(
            all_labels, all_probabilities
        ))
        if len(np.unique(all_labels)) == 2
        else None
    )

    return val_loss, val_accuracy, val_ap

def train_one_fold(train_df, validation_df, device, test_subject, fold_number):
    strongly_labelled = train_df.loc[train_df["label"].astype(int).isin([0, 1])].copy().reset_index(drop=True)
    val_labelled = validation_df.loc[validation_df["label"].astype(int).isin([0, 1])].copy().reset_index(drop=True)
    if strongly_labelled["label"].nunique() < 2:
        raise RuntimeError(f"Fold {test_subject}: strong training labels do not include both classes.")
    if val_labelled.empty:
        raise RuntimeError(f"Fold {test_subject}: no labelled Nathan validation windows.")

    criterion, class_weights = class_weighted_criterion(strongly_labelled["label"].to_numpy(), device)
    loader = DataLoader(
        CachedEEGDataset(strongly_labelled), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    positive_bags = [group.copy() for _, group in train_df.loc[
        train_df["dataset"].eq("sumit") & train_df["recording_label"].eq(1)
    ].groupby("recording_id", sort=True)]
    expected_positive_bags = int(
        train_df.loc[
            train_df["dataset"].eq("sumit")
            & train_df["recording_label"].eq(1),
            "recording_id",
        ].nunique()
    )
    if len(positive_bags) != expected_positive_bags:
        raise RuntimeError(
            f"Expected {expected_positive_bags} Sumit positive recording bags in training; found {len(positive_bags)}."
        )
    for bag in positive_bags:
        if len(bag) == 0 or not bag["label"].astype(int).eq(-1).all():
            raise RuntimeError("A weak positive Sumit bag has invalid instance labels.")

    model = make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_state = None
    best_val_ap = -math.inf
    best_val_loss = math.inf
    best_epoch = 0
    patience_count = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        loss_sum, seen, correct = 0.0, 0, 0
        for batch in loader:
            x = batch["spectrogram"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(y)
            seen += len(y)
            correct += int((logits.detach().argmax(1) == y).sum().item())

        bag_losses = []
        if BAG_LOSS_WEIGHT > 0:
            for _ in range(BAG_REPEATS):
                random.shuffle(positive_bags)
                for bag in positive_bags:
                    bag_loss = run_positive_bag_update(model, optimizer, bag, device)
                    if bag_loss is not None:
                        bag_losses.append(bag_loss)

        val_loss, val_accuracy, val_ap = validation_loss(
    		model, val_labelled, device
	)
        item = {
            "fold_number": fold_number,
            "subject_id": test_subject,
            "epoch": epoch,
            "train_strong_window_loss": loss_sum / max(seen, 1),
            "train_strong_window_accuracy": correct / max(seen, 1),
            "weak_positive_bag_updates": len(bag_losses),
            "mean_weak_positive_bag_loss": float(np.mean(bag_losses)) if bag_losses else None,
            "validation_window_loss": val_loss,
            "validation_window_accuracy": val_accuracy,
	    "validation_average_precision": val_ap,
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(item)
        print(
            f"Epoch {epoch}/{EPOCHS} | supervised "
            f"loss={item['train_strong_window_loss']:.4f} "
            f"acc={item['train_strong_window_accuracy']:.4f} "
            f"| MIL updates={len(bag_losses)} "
            f"| val loss={val_loss:.4f} "
            f"| val acc={'N/A' if val_accuracy is None else f'{val_accuracy:.4f}'} "
            f"| val AP={'N/A' if val_ap is None else f'{val_ap:.4f}'}"
        )
        if val_ap is not None:
            improved = (
                val_ap > best_val_ap + 1e-6
                or (
                    abs(val_ap - best_val_ap) <= 1e-6
                    and val_loss < best_val_loss - 1e-6
                )
            )
        else:
            improved = (
                best_val_ap == -math.inf
                and val_loss < best_val_loss - 1e-6
            )

        if improved:
            if val_ap is not None:
                best_val_ap = val_ap

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_count = 0
        else:
            patience_count += 1

            if patience_count >= PATIENCE:
                print("Early stopping.")
                break
    if best_state is None:
        raise RuntimeError(f"Fold {test_subject}: no best model state was saved.")
    model.load_state_dict(best_state)
    model.to(device)
    return model, history, class_weights, best_epoch, len(positive_bags)


def split_nathan_subjects(nathan_meta, test_subject, fold_number):
    """Deterministic subject-only split shared by both model variants."""
    remaining = nathan_meta.loc[nathan_meta["subject_id"].ne(test_subject)].copy()
    candidates = sorted(remaining["subject_id"].astype(str).unique().tolist())
    if len(candidates) < 2:
        raise RuntimeError("Not enough remaining Nathan subjects for train/validation split.")
    groups = np.asarray(candidates, dtype=object)
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=SEED + fold_number)
    train_indices, val_indices = next(splitter.split(np.zeros((len(groups), 1)), groups=groups))
    train_ids = set(groups[train_indices].tolist())
    val_ids = set(groups[val_indices].tolist())
    train_rows = remaining.loc[remaining["subject_id"].isin(train_ids)].copy()
    val_rows = remaining.loc[remaining["subject_id"].isin(val_ids)].copy()
    return train_rows, val_rows, sorted(train_ids), sorted(val_ids)


def append_csv(path, frame, replace_subject=None):
    if frame is None or frame.empty:
        return
    if path.exists():
        old = pd.read_csv(path)
        if replace_subject is not None and "subject_id" in old.columns:
            old = old.loc[old["subject_id"].astype(str).ne(str(replace_subject))]
        combined = pd.concat([old, frame], ignore_index=True)
    else:
        combined = frame.copy()
    combined.to_csv(path, index=False)


def write_summary(metadata, fold_metrics, device):
    predictions = pd.read_csv(PREDICTIONS_FILE) if PREDICTIONS_FILE.exists() else pd.DataFrame()

    sumit_training_metadata = metadata.loc[
        metadata["dataset"].eq("sumit")
    ].copy()

    if not INCLUDE_SUMIT_TRAINING:
        sumit_training_metadata = sumit_training_metadata.iloc[0:0].copy()

    nathan = metadata.loc[metadata["dataset"].eq("nathan")]
    evaluated = int(fold_metrics["subject_id"].astype(str).nunique()) if not fold_metrics.empty else 0
    total_subjects = int(nathan["subject_id"].nunique())
    summary = {
        "model": "ResNet-18" if MODEL_NAME == "resnet18" else "Pure spatio-temporal EEG Transformer",
        "evaluation_status": "complete_loso" if evaluated == total_subjects else "partial_loso",
        "evaluated_subjects": evaluated,
        "total_nathan_test_subjects": total_subjects,
        "evaluated_nathan_windows": int(len(predictions)),
        "all_dataset_windows": int(len(metadata)),
        "sumit_recordings_in_training": int(sumit_training_metadata["recording_id"].nunique()),
        "sumit_positive_recording_bags": int(sumit_training_metadata.loc[sumit_training_metadata["recording_label"].eq(1), "recording_id"].nunique()),
        "include_sumit_training": INCLUDE_SUMIT_TRAINING,
        "training_method": (
            "Nathan-only supervised window-level cross entropy; Sumit excluded from training"
            if not INCLUDE_SUMIT_TRAINING
            else (
                "Strong window-level cross entropy for Nathan and Sumit-negative windows; MIL disabled for ablation"
                if BAG_LOSS_WEIGHT == 0
                else "Strong window-level cross entropy for Nathan and Sumit-negative windows; max-pooling MIL recording-level loss for positive Sumit recordings"
            )
        ),
        "test_protocol": (
            "Nathan-only subject-level LOSO; Sumit excluded from training."
            if not INCLUDE_SUMIT_TRAINING
            else "Nathan-only subject-level LOSO; all Sumit recordings are in training."
        ),
        "shared_split_rule": "Sorted remaining Nathan subject IDs split with GroupShuffleSplit, validation_size=0.20, random_state=42+fold_number; same runner used for both model variants",
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "mil_bag_repeats_per_epoch": BAG_REPEATS,
        "mil_bag_loss_weight": BAG_LOSS_WEIGHT,
        "resnet_pretrained_imagenet": bool(RESNET_PRETRAINED) if MODEL_NAME == "resnet18" else None,
        "threshold_method": "Per-fold F1 grid search on labelled Nathan validation windows only; unknown Sumit instance labels are excluded",
        "scientific_caveat": "Sumit seizure-positive windows have unknown instance labels. MIL uses recording-level positivity; it does not identify exact seizure windows.",
    }
    if not predictions.empty:
        pooled = calculate_metrics(
            predictions["true_label"].to_numpy(dtype=int),
            predictions["predicted_label"].to_numpy(dtype=int),
            predictions["seizure_probability"].to_numpy(dtype=float),
        )
        summary.update({"pooled_" + key: value for key, value in pooled.items()})
    valid = fold_metrics.copy()
    if not valid.empty:
        both = ((valid["test_tp"] + valid["test_fn"] > 0) & (valid["test_tn"] + valid["test_fp"] > 0))
        valid = valid.loc[both]
    summary["subjects_with_both_test_classes"] = int(len(valid))
    for metric in ["test_sensitivity", "test_specificity", "test_precision", "test_f1", "test_roc_auc", "test_pr_auc"]:
        series = pd.to_numeric(valid[metric], errors="coerce") if (not valid.empty and metric in valid.columns) else pd.Series(dtype=float)
        summary["mean_subject_" + metric.removeprefix("test_")] = float(series.dropna().mean()) if series.notna().any() else None

    POOLED_METRICS_FILE.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"ALL-SUMIT MIL LOSO SUMMARY — {MODEL_NAME.upper()}")
    print("=" * 72)
    print(f"Evaluation status: {summary['evaluation_status']}")
    print(f"Subjects evaluated: {evaluated}/{total_subjects}")
    for key in ["pooled_accuracy", "pooled_sensitivity", "pooled_specificity", "pooled_precision", "pooled_f1", "pooled_roc_auc", "pooled_pr_auc"]:
        value = summary.get(key)
        print(f"{key.removeprefix('pooled_')}: {'N/A' if value is None else f'{value:.4f}'}")
    print(f"Subjects with both classes: {len(valid)}")
    for key in ["mean_subject_sensitivity", "mean_subject_specificity", "mean_subject_precision", "mean_subject_f1", "mean_subject_roc_auc", "mean_subject_pr_auc"]:
        value = summary.get(key)
        print(f"{key}: {'N/A' if value is None else f'{value:.4f}'}")
    print("Saved:", POOLED_METRICS_FILE)


def load_combined_metadata():
    for path in (METADATA_FILE, MANIFEST_FILE, SUMIT_TABLE_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}")
    cache_meta = pd.read_csv(METADATA_FILE)
    manifest = pd.read_csv(MANIFEST_FILE)
    sumit_source = pd.read_csv(SUMIT_TABLE_FILE)

    cache_required = {"cache_file", "epoch_id", "subject_id", "dataset", "label"}
    manifest_required = {"epoch_id", "edf_path", "start_second", "end_second", "label"}
    if cache_required - set(cache_meta.columns):
        raise ValueError(f"STFT cache metadata lacks: {sorted(cache_required - set(cache_meta.columns))}")
    if manifest_required - set(manifest.columns):
        raise ValueError(f"Corrected manifest lacks: {sorted(manifest_required - set(manifest.columns))}")
    if cache_meta["epoch_id"].duplicated().any() or manifest["epoch_id"].duplicated().any():
        raise ValueError("Cache metadata or manifest has duplicate epoch_id values.")

    cache_meta["epoch_id"] = pd.to_numeric(cache_meta["epoch_id"], errors="raise").astype(int)
    manifest["epoch_id"] = pd.to_numeric(manifest["epoch_id"], errors="raise").astype(int)
    cache_meta["label"] = pd.to_numeric(cache_meta["label"], errors="raise").astype(int)
    manifest["label"] = pd.to_numeric(manifest["label"], errors="raise").astype(int)
    if set(cache_meta["epoch_id"]) != set(manifest["epoch_id"]):
        raise ValueError("Cache epoch IDs do not exactly match corrected manifest epoch IDs.")

    extra = manifest[["epoch_id", "edf_path", "start_second", "end_second", "label"]].rename(columns={"label": "manifest_label"})
    metadata = cache_meta.merge(extra, on="epoch_id", how="left", validate="one_to_one")
    if metadata[["edf_path", "start_second", "end_second", "manifest_label"]].isna().any().any():
        raise ValueError("Manifest fields missing after cache-to-manifest join.")
    if not np.array_equal(metadata["label"].to_numpy(), metadata["manifest_label"].astype(int).to_numpy()):
        raise ValueError("Cached labels disagree with the corrected manifest.")
    metadata["dataset"] = metadata["dataset"].astype(str).str.strip().str.lower()
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    metadata["label"] = metadata["label"].astype(int)
    if not set(metadata["dataset"].unique()).issubset({"nathan", "sumit"}):
        raise ValueError("Unexpected dataset name in corrected cache.")
    if not set(metadata["label"].unique()).issubset({-1, 0, 1}):
        raise ValueError("Window labels must be -1 (unknown), 0 (normal), or 1 (seizure).")

    sumit_source["edf_path"] = sumit_source["edf_path"].astype(str)
    sumit_source["label"] = pd.to_numeric(sumit_source["label"], errors="raise").astype(int)
    recording_label_map = sumit_source.set_index("edf_path")["label"].to_dict()
    metadata["recording_id"] = metadata["edf_path"].astype(str)
    metadata["recording_label"] = np.nan
    sumit_mask = metadata["dataset"].eq("sumit")
    metadata.loc[sumit_mask, "recording_label"] = metadata.loc[sumit_mask, "recording_id"].map(recording_label_map)
    if metadata.loc[sumit_mask, "recording_label"].isna().any():
        raise ValueError("Some Sumit cached windows could not be linked to their source recording label.")
    metadata["recording_label"] = pd.to_numeric(metadata["recording_label"], errors="coerce")

    if (metadata.loc[sumit_mask & metadata["recording_label"].eq(0), "label"] != 0).any():
        raise ValueError("A Sumit recording labelled negative contains non-normal window labels.")
    if (metadata.loc[sumit_mask & metadata["recording_label"].eq(1), "label"] != -1).any():
        raise ValueError("Sumit positive recordings must use unknown instance labels (-1).")

    sumit_recordings = metadata.loc[sumit_mask].groupby("recording_id").agg(
        n_windows=("epoch_id", "size"), recording_label=("recording_label", "first")
    )
    if len(sumit_recordings) != 169:
        raise ValueError(f"Expected all 169 Sumit EDF recordings; found {len(sumit_recordings)}.")
    if int((sumit_recordings["recording_label"] == 1).sum()) != 3:
        raise ValueError("Expected exactly 3 Sumit recording-level positive labels in source metadata.")
    if int((sumit_recordings["recording_label"] == 0).sum()) != 166:
        raise ValueError("Expected 166 Sumit recording-level negative labels in source metadata.")
    if (sumit_recordings["n_windows"] != 120).any():
        raise ValueError("Expected 120 full 30-second windows per one-hour Sumit EDF; inspect duration audit.")
    if len(metadata) != 47015:
        raise ValueError(f"Expected 47,015 cached windows; found {len(metadata)}.")
    if len(metadata.loc[metadata["dataset"].eq("sumit") & metadata["label"].eq(-1)]) != 360:
        raise ValueError("Expected 360 Sumit positive windows with unknown instance labels.")
    for path in metadata["cache_file"].astype(str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Cached tensor is missing: {path}")
    return metadata

def write_sumit_lobo_summary(metadata, fold_metrics, device):
    """Summarize held-out Sumit recordings, not individual EEG windows."""
    predictions = (
        pd.read_csv(PREDICTIONS_FILE)
        if PREDICTIONS_FILE.exists()
        else pd.DataFrame()
    )

    sumit = metadata.loc[metadata["dataset"].eq("sumit")]
    total_babies = int(sumit["subject_id"].nunique())
    evaluated_babies = (
        int(fold_metrics["subject_id"].astype(str).nunique())
        if not fold_metrics.empty else 0
    )

    summary = {
        "model": (
            "ResNet-18"
            if MODEL_NAME == "resnet18"
            else "Pure spatio-temporal EEG Transformer"
        ),
        "evaluation_status": (
            "complete_sumit_lobo"
            if evaluated_babies == total_babies
            else "partial_sumit_lobo"
        ),
        "evaluated_babies": evaluated_babies,
        "total_sumit_babies": total_babies,
        "sumit_recordings_tested": int(len(predictions)),
        "evaluation_unit": "recording",
        "recording_score_aggregation": "maximum window seizure probability",
        "training_data": "Nathan plus Sumit recordings from all non-held-out Sumit babies",
        "validation_data": "Nathan validation subjects only",
        "device": str(device),
        "mil_bag_loss_weight": BAG_LOSS_WEIGHT,
        "scientific_caveat": (
            "Only three Sumit recordings are seizure-positive, from two babies. "
            "Recording-level external evaluation therefore has substantial uncertainty."
        ),
    }

    if not predictions.empty:
        pooled = calculate_metrics(
            predictions["true_label"].to_numpy(dtype=int),
            predictions["predicted_label"].to_numpy(dtype=int),
            predictions["seizure_probability"].to_numpy(dtype=float),
        )
        summary.update({
            "recording_level_" + key: value
            for key, value in pooled.items()
        })

    valid = fold_metrics.copy()

    if not valid.empty:
        both_classes = (
            (valid["test_tp"] + valid["test_fn"] > 0)
            & (valid["test_tn"] + valid["test_fp"] > 0)
        )
        valid = valid.loc[both_classes]

    summary["babies_with_both_recording_classes"] = int(len(valid))

    for metric in [
        "sensitivity", "specificity", "precision",
        "f1", "roc_auc", "pr_auc"
    ]:
        column = "test_" + metric
        values = (
            pd.to_numeric(valid[column], errors="coerce").dropna()
            if not valid.empty and column in valid.columns
            else pd.Series(dtype=float)
        )
        summary["mean_baby_" + metric] = (
            float(values.mean()) if not values.empty else None
        )

    POOLED_METRICS_FILE.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== SUMIT RECORDING-LEVEL EVALUATION ===")
    print(f"Babies evaluated: {evaluated_babies}/{total_babies}")
    print(f"Recordings evaluated: {len(predictions)}")
    print("Saved:", POOLED_METRICS_FILE)


def run_sumit_lobo(metadata, nathan_meta, device):
    """Leave one complete Sumit baby's recordings out for testing."""
    sumit_meta = metadata.loc[
        metadata["dataset"].eq("sumit")
    ].copy()

    positive_babies = sorted(
        sumit_meta.loc[
            sumit_meta["recording_label"].eq(1),
            "subject_id"
        ].astype(str).unique().tolist()
    )

    all_babies = sorted(
        sumit_meta["subject_id"].astype(str).unique().tolist()
    )

    # Test the two babies with positive recordings first during the pilot.
    positive_set = set(positive_babies)
    babies = positive_babies + [
        baby for baby in all_babies if baby not in positive_set
    ]

    if len(babies) != 53:
        raise RuntimeError(
            f"Expected 53 Sumit babies; found {len(babies)}."
        )

    if FOLD_METRICS_FILE.exists():
        prior = pd.read_csv(FOLD_METRICS_FILE)
        completed = set(prior["subject_id"].astype(str))
    else:
        prior = pd.DataFrame()
        completed = set()

    nathan_subjects = sorted(
        nathan_meta["subject_id"].astype(str).unique().tolist()
    )

    print("=" * 72)
    print("SUMIT LEAVE-ONE-BABY-OUT EVALUATION")
    print("=" * 72)
    print("Total Sumit babies:", len(babies))
    print("Positive babies:", positive_babies)
    print("Output directory:", OUTPUT_DIR)
    print("MIL loss weight:", BAG_LOSS_WEIGHT)
    print("Folds requested this invocation:", MAX_LOSO_FOLDS)

    completed_this_invocation = 0

    for fold_number, test_baby in enumerate(babies, start=1):
        if fold_number > MAX_LOSO_FOLDS:
            break

        if test_baby in completed:
            print(
                f"[{fold_number}/{len(babies)}] "
                f"Already complete; skipping {test_baby}"
            )
            continue

        print("\n" + "-" * 72)
        print(
            f"FOLD {fold_number}/{len(babies)} "
            f"— held-out Sumit baby: {test_baby}"
        )
        print("-" * 72)

        set_seed(SEED + fold_number)

        # Split Nathan by subject for training and validation.
        groups = np.asarray(nathan_subjects, dtype=object)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_SIZE,
            random_state=SEED + fold_number,
        )

        train_indices, val_indices = next(
            splitter.split(
                np.zeros((len(groups), 1)),
                groups=groups,
            )
        )

        train_ids = set(groups[train_indices].tolist())
        val_ids = set(groups[val_indices].tolist())

        train_nathan_df = nathan_meta.loc[
            nathan_meta["subject_id"].isin(train_ids)
        ].copy()

        validation_df = nathan_meta.loc[
            nathan_meta["subject_id"].isin(val_ids)
        ].copy()

        # Exclude every recording from the held-out Sumit baby.
        test_df = sumit_meta.loc[
            sumit_meta["subject_id"].eq(test_baby)
        ].copy()

        sumit_train_df = sumit_meta.loc[
            sumit_meta["subject_id"].ne(test_baby)
        ].copy()

        train_df = pd.concat(
            [train_nathan_df, sumit_train_df],
            ignore_index=True,
        )

        if set(
            sumit_train_df["subject_id"].astype(str).unique()
        ) & {test_baby}:
            raise RuntimeError(
                "Held-out Sumit baby leaked into training."
            )

        if not validation_df["dataset"].eq("nathan").all():
            raise RuntimeError(
                "Validation must contain Nathan data only."
            )

        print("Nathan training subjects:", len(train_ids))
        print("Nathan validation subjects:", len(val_ids))
        print("Sumit training babies:", sumit_train_df["subject_id"].nunique())
        print("Held-out Sumit recordings:", test_df["recording_id"].nunique())

        model, history, class_weights, best_epoch, bag_count = train_one_fold(
            train_df,
            validation_df,
            device,
            test_baby,
            fold_number,
        )

        # Select the threshold on Nathan validation recordings.
        val_predictions = collect_predictions(
            model, validation_df, device
        )

        val_paths = validation_df[
            ["epoch_id", "edf_path"]
        ].copy()

        val_joined = val_predictions.merge(
            val_paths,
            on="epoch_id",
            how="left",
            validate="one_to_one",
        )

        val_recordings = val_joined.groupby(
            "edf_path", as_index=False
        ).agg(
            true_label=("true_label", "max"),
            seizure_probability=("seizure_probability", "max"),
        )

        threshold, validation_f1, threshold_method = select_sumit_recording_threshold(
        val_recordings["true_label"].to_numpy(dtype=int),
        val_recordings["seizure_probability"].to_numpy(dtype=float),
        )
        # Predict all windows, including Sumit's unknown-label windows.
        window_predictions = collect_predictions(
            model,
            test_df,
            device,
            include_unknown=True,
        )

        mapping = test_df[
            ["epoch_id", "recording_id", "recording_label"]
        ].copy()

        joined = window_predictions.merge(
            mapping,
            on="epoch_id",
            how="left",
            validate="one_to_one",
        )

        # One maximum-probability score for each held-out recording.
        recording_predictions = joined.groupby(
            ["subject_id", "recording_id", "recording_label"],
            as_index=False,
        ).agg(
            seizure_probability=("seizure_probability", "max"),
            n_windows=("epoch_id", "nunique"),
        )

        recording_predictions["true_label"] = (
            recording_predictions["recording_label"].astype(int)
        )
        recording_predictions["predicted_label"] = (
            recording_predictions["seizure_probability"] >= threshold
        ).astype(int)
        recording_predictions["threshold_used"] = threshold

        metrics = calculate_metrics(
            recording_predictions["true_label"].to_numpy(dtype=int),
            recording_predictions["predicted_label"].to_numpy(dtype=int),
            recording_predictions["seizure_probability"].to_numpy(dtype=float),
        )

        # Save per-recording predictions; never save them as window labels.
        append_csv(
            PREDICTIONS_FILE,
            recording_predictions,
            replace_subject=test_baby,
        )
        append_csv(
            HISTORY_FILE,
            pd.DataFrame(history),
            replace_subject=test_baby,
        )

        fold_row = {
            "fold_number": fold_number,
            "subject_id": test_baby,
            "model": MODEL_NAME,
            "nathan_train_subjects": len(train_ids),
            "nathan_validation_subjects": len(val_ids),
            "sumit_babies_in_training": int(
                sumit_train_df["subject_id"].nunique()
            ),
            "sumit_recordings_in_training": int(
                sumit_train_df["recording_id"].nunique()
            ),
            "sumit_positive_bags_in_training": bag_count,
            "test_recordings": len(recording_predictions),
            "test_windows": len(test_df),
            "best_epoch": best_epoch,
            "validation_threshold": threshold,
            "validation_f1_at_threshold": validation_f1,
            "threshold_method": threshold_method,
            "training_seconds": None,
            "class_weight_normal": class_weights[0],
            "class_weight_seizure": class_weights[1],
        }

        for key, value in metrics.items():
            fold_row[
                "test_n_epochs" if key == "n_epochs" else "test_" + key
            ] = value

        if FOLD_METRICS_FILE.exists():
            existing = pd.read_csv(FOLD_METRICS_FILE)
            existing = existing.loc[
                existing["subject_id"].astype(str).ne(test_baby)
            ]
            updated = pd.concat(
                [existing, pd.DataFrame([fold_row])],
                ignore_index=True,
            )
        else:
            updated = pd.DataFrame([fold_row])

        updated = updated.sort_values("fold_number").reset_index(drop=True)
        updated.to_csv(FOLD_METRICS_FILE, index=False)

        completed.add(test_baby)
        completed_this_invocation += 1

        print(
            f"Threshold={threshold:.2f}; "
            f"recordings tested={len(recording_predictions)}; "
            f"recording ROC-AUC={metrics['roc_auc']}; "
            f"recording PR-AUC={metrics['pr_auc']}"
        )

        write_sumit_lobo_summary(metadata, updated, device)

    final_folds = (
        pd.read_csv(FOLD_METRICS_FILE)
        if FOLD_METRICS_FILE.exists()
        else pd.DataFrame()
    )

    write_sumit_lobo_summary(metadata, final_folds, device)

    print("\nSumit evaluation finished.")
    print("New folds completed:", completed_this_invocation)
    print("Fold metrics:", FOLD_METRICS_FILE)
    print("Recording predictions:", PREDICTIONS_FILE)
    print("Summary:", POOLED_METRICS_FILE)
def main():
    if MAX_LOSO_FOLDS < 1 or BATCH_SIZE < 1 or EPOCHS < 1 or PATIENCE < 1 or BAG_REPEATS < 1:
        raise ValueError("Fold, batch, epoch, patience, and MIL-repeat values must be positive.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = get_device()
    metadata = load_combined_metadata()
    nathan_meta = metadata.loc[metadata["dataset"].eq("nathan")].copy()
    evaluation_mode = os.environ.get(
        "EEG_EVAL_MODE", "nathan_loso"
    ).strip().lower()

    if evaluation_mode == "sumit_lobo":
        run_sumit_lobo(metadata, nathan_meta, device)
        return

    if evaluation_mode != "nathan_loso":
        raise ValueError(
            "EEG_EVAL_MODE must be 'nathan_loso' or 'sumit_lobo'."
        )
    subjects = sorted(nathan_meta["subject_id"].unique().tolist())
    if len(subjects) != 79:
        raise ValueError(f"Expected 79 Nathan test subjects; found {len(subjects)}.")

    print("=" * 72)
    if INCLUDE_SUMIT_TRAINING:
        print(f"ALL 169 SUMIT RECORDINGS + NATHAN ? MATCHED-SPLIT LOSO ({MODEL_NAME.upper()})")
    else:
        print(f"NATHAN-ONLY ? MATCHED-SPLIT LOSO ({MODEL_NAME.upper()})")
    print("=" * 72)
    print("Device:", device)
    print("Cache:", METADATA_FILE)
    print("Model output:", OUTPUT_DIR)
    print("Total windows:", f"{len(metadata):,}")
    print("Window labels:", metadata["label"].value_counts().sort_index().to_dict())
    print("Sumit recordings:", metadata.loc[metadata["dataset"].eq("sumit"), "recording_id"].nunique())
    print("Sumit recording labels:", metadata.loc[metadata["dataset"].eq("sumit")].drop_duplicates("recording_id")["recording_label"].value_counts().sort_index().to_dict())
    if INCLUDE_SUMIT_TRAINING:
        print("Combined training uses Nathan and Sumit.")
    else:
        print("Nathan-only training; Sumit excluded.")
    print("Validation and held-out test subjects: Nathan only; splits reproducible across model variants.")
    print("Batch:", BATCH_SIZE, "Epochs:", EPOCHS, "Patience:", PATIENCE, "MIL repeats:", BAG_REPEATS)
    print("Pretrained ResNet-18:", RESNET_PRETRAINED if MODEL_NAME == "resnet18" else "not applicable")
    print("Outer folds this invocation:", MAX_LOSO_FOLDS)

    if FOLD_METRICS_FILE.exists():
        prior = pd.read_csv(FOLD_METRICS_FILE)
        completed = set(prior["subject_id"].astype(str)) if "subject_id" in prior else set()
    else:
        prior = pd.DataFrame()
        completed = set()

    completed_this_invocation = 0
    for fold_number, test_subject in enumerate(subjects, start=1):
        if fold_number > MAX_LOSO_FOLDS:
            break
        if test_subject in completed:
            print(f"[{fold_number}/{len(subjects)}] Already complete; skipping {test_subject}")
            continue
        print("\n" + "-" * 72)
        print(f"FOLD {fold_number}/{len(subjects)} — held-out Nathan subject: {test_subject}")
        print("-" * 72)
        set_seed(SEED + fold_number)

        test_df = nathan_meta.loc[nathan_meta["subject_id"].eq(test_subject)].copy().reset_index(drop=True)
        train_nathan_df, validation_df, train_nathan_subjects, validation_subjects = split_nathan_subjects(
            nathan_meta, test_subject, fold_number
        )
        available_sumit_df = metadata.loc[
            metadata["dataset"].eq("sumit")
        ].copy()

        sumit_train_df = (
            available_sumit_df
            if INCLUDE_SUMIT_TRAINING
            else available_sumit_df.iloc[0:0].copy()
        )

        train_df = pd.concat(
            [train_nathan_df, sumit_train_df],
            ignore_index=True
        )

        # Sumit is included only when requested; validation and test remain Nathan-only.
        if not validation_df["dataset"].eq("nathan").all() or not test_df["dataset"].eq("nathan").all():
            raise RuntimeError("Validation/test sets must contain Nathan data only.")
        if set(train_nathan_subjects) & set(validation_subjects) or test_subject in set(train_nathan_subjects) | set(validation_subjects):
            raise RuntimeError("Nathan subject leakage detected in shared split.")
        if INCLUDE_SUMIT_TRAINING and len(sumit_train_df["recording_id"].unique()) != 169:
            raise RuntimeError("Not all Sumit recordings are represented in training.")

        if not INCLUDE_SUMIT_TRAINING and not sumit_train_df.empty:
            raise RuntimeError("Sumit data unexpectedly entered Nathan-only training.")
        strong_train = train_df.loc[train_df["label"].isin([0, 1])]
        print(f"Windows — train total {len(train_df):,}; supervised {len(strong_train):,}; validation {len(validation_df):,}; test {len(test_df):,}")
        print(f"Subjects ? Nathan train {len(train_nathan_subjects)}, validation {len(validation_subjects)}, test 1")
        print("Sumit recordings in train:", sumit_train_df["recording_id"].nunique())
        print(f"Validation subjects: {','.join(validation_subjects)}")

        started = time.time()
        model, history, class_weights, best_epoch, bag_count = train_one_fold(
            train_df, validation_df, device, test_subject, fold_number
        )
        val_predictions = collect_predictions(model, validation_df, device)
        threshold, validation_f1, threshold_method = select_threshold(
            val_predictions["true_label"].to_numpy(dtype=int),
            val_predictions["seizure_probability"].to_numpy(dtype=float),
        )
        test_predictions = collect_predictions(model, test_df, device)
        test_predictions["predicted_label"] = (test_predictions["seizure_probability"] >= threshold).astype(int)
        test_predictions["threshold_used"] = threshold
        metrics = calculate_metrics(
            test_predictions["true_label"].to_numpy(dtype=int),
            test_predictions["predicted_label"].to_numpy(dtype=int),
            test_predictions["seizure_probability"].to_numpy(dtype=float),
        )

        append_csv(PREDICTIONS_FILE, test_predictions, replace_subject=test_subject)
        append_csv(HISTORY_FILE, pd.DataFrame(history), replace_subject=test_subject)
        fold_row = {
            "fold_number": fold_number,
            "subject_id": test_subject,
            "model": MODEL_NAME,
            "nathan_train_subjects": len(train_nathan_subjects),
            "validation_subjects_count": len(validation_subjects),
            "validation_subjects": ",".join(validation_subjects),
            "sumit_recordings_in_train": len(sumit_train_df["recording_id"].unique()),
            "train_total_windows": len(train_df),
            "train_supervised_windows": len(strong_train),
            "validation_windows": len(validation_df),
            "test_windows": len(test_df),
            "sumit_positive_bags": bag_count,
            "mil_updates_per_epoch": bag_count * BAG_REPEATS if BAG_LOSS_WEIGHT > 0 else 0,
            "best_epoch": best_epoch,
            "validation_threshold": threshold,
            "validation_f1_at_threshold": validation_f1,
            "threshold_method": threshold_method,
            "training_seconds": round(time.time() - started, 2),
            "class_weight_normal": class_weights[0],
            "class_weight_seizure": class_weights[1],
        }
        for key, value in metrics.items():
            fold_row["test_n_epochs" if key == "n_epochs" else "test_" + key] = value
        if FOLD_METRICS_FILE.exists():
            existing = pd.read_csv(FOLD_METRICS_FILE)
            existing = existing.loc[existing["subject_id"].astype(str).ne(test_subject)]
            updated = pd.concat([existing, pd.DataFrame([fold_row])], ignore_index=True)
        else:
            updated = pd.DataFrame([fold_row])
        updated = updated.sort_values("fold_number").reset_index(drop=True)
        updated.to_csv(FOLD_METRICS_FILE, index=False)
        prior = updated
        completed.add(test_subject)
        completed_this_invocation += 1

        roc = metrics["roc_auc"]
        pr = metrics["pr_auc"]
        sens = metrics["sensitivity"]
        spec = metrics["specificity"]
        print(
            f"Threshold={threshold:.2f}; test accuracy={metrics['accuracy']:.4f}, "
            f"sensitivity={'N/A' if sens is None else f'{sens:.4f}'}, "
            f"specificity={'N/A' if spec is None else f'{spec:.4f}'}, "
            f"F1={metrics['f1']:.4f}, "
            f"ROC-AUC={'N/A' if roc is None else f'{roc:.4f}'}, "
            f"PR-AUC={'N/A' if pr is None else f'{pr:.4f}'}"
        )
        write_summary(metadata, updated, device)

    final_folds = pd.read_csv(FOLD_METRICS_FILE) if FOLD_METRICS_FILE.exists() else pd.DataFrame()
    write_summary(metadata, final_folds, device)
    print("\nRun finished.")
    print("New folds completed this invocation:", completed_this_invocation)
    print("Fold metrics:", FOLD_METRICS_FILE)
    print("Predictions:", PREDICTIONS_FILE)
    print("Training history:", HISTORY_FILE)
    print("Pooled summary:", POOLED_METRICS_FILE)


if __name__ == "__main__":
    main()
