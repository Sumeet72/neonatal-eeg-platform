"""
Generate remaining final evaluation graphs (4-8) for the neonatal EEG project.

Run from project root:
    python -m scripts.generate_remaining_graphs

This script does NOT retrain anything and does NOT rebuild the STFT cache.
Graphs 1-3 are already generated, so this script only creates 4-8.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
PRED = ROOT / "results" / "ResNet18_evaluation" / "ResNet18_all_LOSO_predictions.csv"
HIST = ROOT / "results" / "final_model" / "ResNet18_final_training_history.csv"
OUT = ROOT / "results" / "final_graphs"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.40

def save(fig, name):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}", flush=True)

print("=" * 60, flush=True)
print("Generating remaining final graphs (4-8)", flush=True)
print("=" * 60, flush=True)

# ------------------------------------------------------------
# Load LOSO predictions
# ------------------------------------------------------------
if not PRED.exists():
    raise FileNotFoundError(f"Missing predictions file: {PRED}")

df = pd.read_csv(PRED)
print(f"Loaded predictions: {len(df):,} rows", flush=True)

# Robust column detection
def find_col(candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in df.columns:
        cl = c.lower()
        for candidate in candidates:
            if candidate.lower() in cl:
                return c
    return None

label_col = find_col(["label", "true_label", "y_true", "target"])
prob_col = find_col([
    "seizure_probability", "probability", "seizure_prob",
    "prob", "positive_probability"
])

if label_col is None or prob_col is None:
    raise ValueError(
        f"Could not identify label/probability columns.\n"
        f"Columns found: {list(df.columns)}"
    )

y = df[label_col].astype(int).to_numpy()
p = df[prob_col].astype(float).to_numpy()

# ------------------------------------------------------------
# Graph 4: Sensitivity / Specificity vs threshold
# ------------------------------------------------------------
thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
rows = []

for t in thresholds:
    pred = (p >= t).astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity > 0
        else 0.0
    )

    rows.append({
        "threshold": t,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    })

tm = pd.DataFrame(rows)
tm.to_csv(OUT / "threshold_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tm["threshold"], tm["sensitivity"], marker="o", label="Sensitivity")
ax.plot(tm["threshold"], tm["specificity"], marker="o", label="Specificity")
ax.axvline(THRESHOLD, linestyle="--", label="Operating threshold = 0.40")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Sensitivity and Specificity vs Decision Threshold")
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)
ax.legend()
save(fig, "04_sensitivity_specificity_vs_threshold.png")

        # ------------------------------------------------------------
# Graph 5: F1 vs threshold
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tm["threshold"], tm["f1"], marker="o", label="F1-score")
current_f1 = float(tm.loc[np.isclose(tm["threshold"], THRESHOLD), "f1"].iloc[0])
ax.axvline(THRESHOLD, linestyle="--", label=f"Operating threshold = 0.40")
ax.scatter([THRESHOLD], [current_f1], s=70, zorder=3)
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("F1-score")
ax.set_title("F1-score vs Decision Threshold")
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)
ax.legend()
save(fig, "05_f1_vs_threshold.png")
# ------------------------------------------------------------
# Graph 6: CNN vs ResNet-18
# ------------------------------------------------------------
comparison = pd.DataFrame({
    "Metric": [
        "Accuracy", "Precision", "Sensitivity",
        "Specificity", "F1-score", "ROC-AUC"
    ],
    "CNN": [0.7639, 0.2736, 0.5249, 0.7985, 0.3597, 0.7191],
    "ResNet18": [0.8055, 0.3476, 0.6155, 0.8329, 0.4443, 0.8153]
})
comparison.to_csv(OUT / "model_comparison.csv", index=False)

x = np.arange(len(comparison))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - width/2, comparison["CNN"], width, label="Baseline CNN")
ax.bar(x + width/2, comparison["ResNet18"], width, label="ResNet-18")
ax.set_xticks(x)
ax.set_xticklabels(comparison["Metric"], rotation=20, ha="right")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("Baseline CNN vs Adapted ResNet-18")
ax.grid(axis="y", alpha=0.25)
ax.legend()
save(fig, "06_CNN_vs_ResNet18.png")

# ------------------------------------------------------------
# Graphs 7-8: final training history
# ------------------------------------------------------------
if not HIST.exists():
    print(f"WARNING: Training history not found: {HIST}", flush=True)
    print("Graphs 7-8 were not generated.", flush=True)
else:
    hist = pd.read_csv(HIST)
    print(f"Loaded training history: {len(hist)} rows", flush=True)
    print(f"History columns: {list(hist.columns)}", flush=True)

    def history_col(kind):
        # Prefer exact common names, then substring matching.
        candidates = {
            "loss": ["train_loss", "loss", "training_loss"],
            "accuracy": ["train_accuracy", "accuracy", "train_acc", "acc", "training_accuracy"]
        }[kind]
        lower = {c.lower(): c for c in hist.columns}
        for c in candidates:
            if c.lower() in lower:
                return lower[c.lower()]
        for c in hist.columns:
            cl = c.lower()
            if kind == "loss" and "loss" in cl:
                return c
            if kind == "accuracy" and ("accuracy" in cl or cl.endswith("acc")):
                return c
        return None

    epoch_col = None
    for c in hist.columns:
        if c.lower() in ("epoch", "epochs"):
            epoch_col = c
            break
    if epoch_col is None:
        epochs = np.arange(len(hist))
    else:
        epochs = hist[epoch_col].to_numpy()

    loss_col = history_col("loss")
    acc_col = history_col("accuracy")

    if loss_col is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, hist[loss_col], marker="o", label="Training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("ResNet-18 Final Training Loss")
        ax.grid(alpha=0.25)
        ax.legend()
        save(fig, "07_training_loss.png")
    else:
        print("WARNING: Could not identify training-loss column.", flush=True)

    if acc_col is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, hist[acc_col], marker="o", label="Training accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("ResNet-18 Final Training Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()
        save(fig, "08_training_accuracy.png")
    else:
        print("WARNING: Could not identify training-accuracy column.", flush=True)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
summary = {
    "operating_threshold": THRESHOLD,
    "graphs_generated": sorted(
        p.name for p in OUT.glob("0*.png")
        if p.name.startswith(("04_", "05_", "06_", "07_", "08_"))
    ),
    "note": (
        "The 0.40 threshold is retained as the pre-specified operating "
        "threshold. The pooled-test threshold with maximum F1 is not used "
        "as the final operating threshold because selecting it from held-out "
        "test predictions would introduce test-set leakage."
    )
}
with open(OUT / "remaining_graph_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("=" * 60, flush=True)
print("DONE", flush=True)
print(f"Output folder: {OUT}", flush=True)
print("=" * 60, flush=True)
