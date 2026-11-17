"""ResNet-18 LOSO baseline on the corrected mixed Nathan + Sumit cache.

Test folds are Nathan subjects only. Sumit negative windows and remaining
Nathan subjects may participate in train/validation, with subject-level split.
This is a standalone comparison runner; it does not alter other experiments.
"""

import json
import os
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
from src.models.ResNet18 import ResNet18EEG
from src.training.losses import get_weighted_loss
from src.training.trainer import train_model
from src.utils.device import get_device
from src.utils.seed import set_seed

SEED = 42
METADATA_FILE = Path(os.environ.get(
    "RESNET18_COMPARE_METADATA_FILE",
    "results/corrected_epoch_dataset/stft_cache/metadata.csv",
))
OUTPUT_DIR = Path(os.environ.get(
    "RESNET18_COMPARE_OUTPUT_DIR",
    "results/ResNet18_corrected_Nathan_LOSO",
))
PREDICTION_DIR = OUTPUT_DIR / "predictions"
HISTORY_DIR = OUTPUT_DIR / "training_history"
FOLD_METRICS_FILE = OUTPUT_DIR / "fold_metrics.csv"
POOLED_METRICS_FILE = OUTPUT_DIR / "pooled_metrics.json"

BATCH_SIZE = int(os.environ.get("RESNET18_COMPARE_BATCH_SIZE", "8"))
EPOCHS = int(os.environ.get("RESNET18_COMPARE_EPOCHS", "5"))
PATIENCE = int(os.environ.get("RESNET18_COMPARE_PATIENCE", "2"))
LEARNING_RATE = float(os.environ.get("RESNET18_COMPARE_LEARNING_RATE", "0.0001"))
MAX_LOSO_FOLDS = int(os.environ.get("MAX_RESNET18_COMPARE_FOLDS", "1"))
VALIDATION_SIZE = 0.20
THRESHOLD_GRID = np.round(np.arange(0.10, 0.901, 0.05), 2)
PRETRAINED = os.environ.get("RESNET18_COMPARE_PRETRAINED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}


def calculate_metrics(y_true, y_pred, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "n_epochs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(y_true, probabilities)) if has_both_classes else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_threshold_from_validation(y_true, probabilities):
    """Select threshold on validation subjects only, maximizing validation F1."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) == 0 or not np.any(y_true == 1):
        return 0.40, 0.0, "fallback_0.40_no_validation_positives"

    best_threshold, best_f1 = 0.40, -1.0
    for threshold in THRESHOLD_GRID:
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1, "validation_f1_grid_search"


@torch.inference_mode()
def collect_predictions(model, loader, device):
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


def history_frame(history):
    if isinstance(history, pd.DataFrame):
        return history
    if isinstance(history, dict):
        try:
            return pd.DataFrame(history)
        except Exception:
            return pd.DataFrame({"history": [str(history)]})
    try:
        return pd.DataFrame(history)
    except Exception:
        return pd.DataFrame({"history": [str(history)]})


def write_summary(metadata, folds, device):
    prediction_files = sorted(PREDICTION_DIR.glob("*.csv"))
    prediction_frames = [pd.read_csv(path) for path in prediction_files]
    prediction_frames = [frame for frame in prediction_frames if not frame.empty]
    combined = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    total_subjects = int(
        metadata.loc[metadata["dataset"].str.lower().eq("nathan"), "subject_id"].nunique()
    )
    evaluated_subjects = int(folds["subject_id"].astype(str).nunique()) if not folds.empty else 0

    summary = {
        "model": "ResNet-18 EEG baseline",
        "evaluation_status": "complete_loso" if evaluated_subjects == total_subjects else "partial_loso_pilot",
        "evaluated_subjects": evaluated_subjects,
        "total_subjects": total_subjects,
        "evaluated_epochs": int(len(combined)),
        "dataset_epoch_count": int(len(metadata)),
        "test_dataset": "nathan",
        "training_data": "Nathan plus eligible seizure-negative Sumit windows; held-out subjects are Nathan only",
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "pretrained_imagenet_weights": bool(PRETRAINED),
        "threshold_method": "Per-fold threshold selected from validation subjects only by F1 grid search; fallback 0.40 when validation has no positives",
    }

    if not combined.empty:
        pooled = calculate_metrics(
            combined["true_label"].to_numpy(dtype=int),
            combined["predicted_label"].to_numpy(dtype=int),
            combined["seizure_probability"].to_numpy(dtype=float),
        )
        summary.update(pooled)

    # Macro mean over test subjects whose test set contains both classes.
    valid = folds.copy()
    if not valid.empty:
        both_classes = (valid["test_tp"] + valid["test_fn"] > 0) & (valid["test_tn"] + valid["test_fp"] > 0)
        valid = valid.loc[both_classes]
    summary["subjects_with_both_test_classes"] = int(len(valid))
    for metric in ["test_sensitivity", "test_specificity", "test_precision", "test_f1", "test_roc_auc", "test_pr_auc"]:
        summary["mean_subject_" + metric.removeprefix("test_")] = (
            float(valid[metric].mean()) if not valid.empty and metric in valid else None
        )

    with POOLED_METRICS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    print("\n" + "=" * 72)
    print("RESNET-18 CORRECTED-DATA LOSO SUMMARY")
    print("=" * 72)
    print(f"Evaluation status: {summary['evaluation_status']}")
    print(f"Subjects evaluated: {evaluated_subjects}/{total_subjects}")
    print(f"Epoch predictions: {len(combined):,}")
    for key in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "pr_auc"):
        value = summary.get(key)
        print(f"{key}: {'N/A' if value is None else f'{value:.4f}'}")
    print(f"Subjects with both test classes: {summary['subjects_with_both_test_classes']}")
    print("Mean subject metrics (subjects with both classes):")
    for key in ("mean_subject_sensitivity", "mean_subject_specificity", "mean_subject_precision", "mean_subject_f1", "mean_subject_roc_auc", "mean_subject_pr_auc"):
        value = summary.get(key)
        print(f"  {key}: {'N/A' if value is None else f'{value:.4f}'}")
    print("Saved:", POOLED_METRICS_FILE)
    if evaluated_subjects != total_subjects:
        print("NOTE: Partial evaluation only; rerun with a larger MAX_RESNET18_COMPARE_FOLDS to resume.")


def main():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Corrected STFT metadata not found: {METADATA_FILE}")
    if MAX_LOSO_FOLDS < 1:
        raise ValueError("MAX_RESNET18_COMPARE_FOLDS must be at least 1")
    if BATCH_SIZE < 1 or EPOCHS < 1 or PATIENCE < 1:
        raise ValueError("Batch size, epochs, and patience must be positive")

    for directory in (OUTPUT_DIR, PREDICTION_DIR, HISTORY_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = get_device()
    metadata = pd.read_csv(METADATA_FILE)
    required = {"cache_file", "epoch_id", "subject_id", "dataset", "label"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Corrected cache metadata missing columns: {sorted(missing)}")
    if metadata[list(required)].isna().any().any():
        raise ValueError("Corrected cache metadata contains missing required values")

    metadata["subject_id"] = metadata["subject_id"].astype(str)
    metadata["dataset"] = metadata["dataset"].astype(str)
    metadata["label"] = metadata["label"].astype(int)
    if not set(metadata["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0=normal, 1=seizure")

    test_metadata = metadata[metadata["dataset"].str.lower().eq("nathan")].copy()
    if test_metadata.empty:
        raise ValueError("No Nathan subjects were found in the corrected cache")
    subjects = sorted(test_metadata["subject_id"].unique().tolist())

    print("=" * 72)
    print("RESNET-18 BASELINE — CORRECTED MIXED DATA, NATHAN SUBJECT-LEVEL LOSO")
    print("=" * 72)
    print("Device:", device)
    print("Metadata:", METADATA_FILE)
    print("Output:", OUTPUT_DIR)
    print("All training/validation windows:", len(metadata))
    print("Dataset counts:\n", metadata.groupby(["dataset", "label"]).size().to_string())
    print("Nathan test subjects:", len(subjects))
    print("Batch size:", BATCH_SIZE, "| Epochs/fold:", EPOCHS, "| Patience:", PATIENCE)
    print("Pretrained ImageNet weights:", PRETRAINED)
    print("Outer folds allowed in this invocation:", MAX_LOSO_FOLDS)
    print("The original STFT cache, Transformer results, and ResNet checkpoint will not be modified.")

    if FOLD_METRICS_FILE.exists():
        prior = pd.read_csv(FOLD_METRICS_FILE)
        completed_subjects = set(prior["subject_id"].astype(str)) if "subject_id" in prior.columns else set()
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

        print("\n" + "-" * 72)
        print(f"FOLD {fold_number}/{len(subjects)} — held-out Nathan subject: {test_subject}")
        print("-" * 72)
        set_seed(SEED + fold_number)

        test_df = test_metadata[test_metadata["subject_id"].eq(test_subject)].copy().reset_index(drop=True)
        remaining_df = metadata[metadata["subject_id"].ne(test_subject)].copy().reset_index(drop=True)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_SIZE,
            random_state=SEED + fold_number,
        )
        train_idx, validation_idx = next(
            splitter.split(remaining_df, remaining_df["label"], groups=remaining_df["subject_id"])
        )
        train_df = remaining_df.iloc[train_idx].copy().reset_index(drop=True)
        validation_df = remaining_df.iloc[validation_idx].copy().reset_index(drop=True)

        train_subjects = set(train_df["subject_id"].astype(str))
        validation_subjects = set(validation_df["subject_id"].astype(str))
        test_subjects = set(test_df["subject_id"].astype(str))
        if train_subjects & validation_subjects or train_subjects & test_subjects or validation_subjects & test_subjects:
            raise RuntimeError("Subject leakage detected in train/validation/test split")
        if train_df["label"].nunique() < 2:
            raise RuntimeError(f"Training split for {test_subject} has only one class")

        print(f"Epochs — train: {len(train_df)}, validation: {len(validation_df)}, test: {len(test_df)}")
        print(f"Subjects — train: {len(train_subjects)}, validation: {len(validation_subjects)}, test: 1")

        pin_memory = device.type == "cuda"
        train_loader = DataLoader(CachedEEGDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
        validation_loader = DataLoader(CachedEEGDataset(validation_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)
        test_loader = DataLoader(CachedEEGDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

        model = ResNet18EEG(num_channels=9, num_classes=2, pretrained=PRETRAINED).to(device)
        criterion, class_weights = get_weighted_loss(train_df["label"].to_numpy(copy=True), device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        start_time = time.time()
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
        training_seconds = time.time() - start_time

        validation_output = collect_predictions(model, validation_loader, device)
        threshold, validation_f1, threshold_method = select_threshold_from_validation(
            validation_output["true_label"], validation_output["seizure_probability"]
        )
        test_output = collect_predictions(model, test_loader, device)
        probabilities = test_output["seizure_probability"]
        predicted = (probabilities >= threshold).astype(int)
        test_metrics = calculate_metrics(test_output["true_label"], predicted, probabilities)

        prediction_df = pd.DataFrame({
            "subject_id": test_output["subject_id"],
            "dataset": test_output["dataset"],
            "epoch_id": test_output["epoch_id"],
            "true_label": test_output["true_label"],
            "seizure_probability": probabilities,
            "threshold_used": threshold,
            "predicted_label": predicted,
        })
        safe_subject = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in test_subject)
        prediction_df.to_csv(PREDICTION_DIR / f"{safe_subject}.csv", index=False)
        history_frame(history).to_csv(HISTORY_DIR / f"{safe_subject}.csv", index=False)

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
            "training_seconds": round(training_seconds, 2),
            "class_weight_normal": float(class_weights[0].detach().cpu()) if torch.is_tensor(class_weights) else None,
            "class_weight_seizure": float(class_weights[1].detach().cpu()) if torch.is_tensor(class_weights) else None,
        }
        for key, value in test_metrics.items():
            if key != "n_epochs":
                fold_row["test_" + key] = value
        fold_row["test_n_epochs"] = test_metrics["n_epochs"]

        updated = pd.concat([prior, pd.DataFrame([fold_row])], ignore_index=True)
        updated = updated.drop_duplicates(subset=["subject_id"], keep="last")
        updated = updated.sort_values("fold_number").reset_index(drop=True)
        updated.to_csv(FOLD_METRICS_FILE, index=False)
        prior = updated
        completed_subjects.add(test_subject)
        completed_this_invocation += 1

        print(f"Threshold selected on validation: {threshold:.2f} (validation F1={validation_f1:.4f})")
        roc_text = "N/A" if test_metrics["roc_auc"] is None else f"{test_metrics['roc_auc']:.4f}"
        pr_text = "N/A" if test_metrics["pr_auc"] is None else f"{test_metrics['pr_auc']:.4f}"
        print(
            "Test metrics: "
            f"accuracy={test_metrics['accuracy']:.4f}, "
            f"sensitivity={test_metrics['sensitivity']:.4f}, "
            f"specificity={test_metrics['specificity']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, "
            f"ROC-AUC={roc_text}, PR-AUC={pr_text}"
        )
        print("Saved fold predictions and metrics.")
        write_summary(metadata, prior, device)

    # Write one final summary even when this invocation only skipped finished folds.
    final_folds = pd.read_csv(FOLD_METRICS_FILE) if FOLD_METRICS_FILE.exists() else pd.DataFrame()
    write_summary(metadata, final_folds, device)
    print("\nRun finished.")
    print("New folds completed this invocation:", completed_this_invocation)
    print("Fold metrics:", FOLD_METRICS_FILE)
    print("Pooled summary:", POOLED_METRICS_FILE)


if __name__ == "__main__":
    main()
