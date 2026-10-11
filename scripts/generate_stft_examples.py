"""
Generate Graphs 9-10: real Normal and Seizure STFT examples.

Run from project root:
    python -m scripts.generate_stft_examples

Uses the existing STFT cache. No retraining and no preprocessing rebuild.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path.cwd()
META = ROOT / "results" / "stft_cache" / "metadata.csv"
OUT = ROOT / "results" / "final_graphs"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2", "T3", "T4", "Cz"]
FS = 256
N_FFT = 256
HOP = 128
FREQS = np.arange(N_FFT // 2 + 1) * FS / N_FFT
TIMES = (np.arange(59) * HOP + N_FFT / 2) / FS

print("=" * 60, flush=True)
print("Generating real STFT examples (9-10)", flush=True)
print("=" * 60, flush=True)

if not META.exists():
    raise FileNotFoundError(f"Missing STFT metadata: {META}")

meta = pd.read_csv(META)
print(f"Loaded cache metadata: {len(meta):,} rows", flush=True)
print(f"Columns: {list(meta.columns)}", flush=True)

def find_col(candidates):
    lower = {c.lower(): c for c in meta.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in meta.columns:
        cl = c.lower()
        for candidate in candidates:
            if candidate.lower() in cl:
                return c
    return None

label_col = find_col(["label", "true_label", "target"])
subject_col = find_col(["subject_id", "subject", "patient_id"])
cache_col = find_col(["cache_file", "cache_path", "path", "file"])
epoch_col = find_col(["epoch_id", "epoch"])

if label_col is None or cache_col is None:
    raise ValueError(
        "Could not identify required metadata columns. "
        f"Columns: {list(meta.columns)}"
    )

print(
    f"Using columns: label={label_col}, cache={cache_col}, "
    f"subject={subject_col}, epoch={epoch_col}",

flush=True
)

# Prefer N001 because it was already used in the real inference sanity check.
normal = meta[meta[label_col].astype(int) == 0]
seizure = meta[meta[label_col].astype(int) == 1]

if subject_col is not None:
    n001_normal = normal[normal[subject_col].astype(str) == "N001"]
    n001_seizure = seizure[seizure[subject_col].astype(str) == "N001"]
    if len(n001_normal):
        normal = n001_normal
    if len(n001_seizure):
        seizure = n001_seizure

if len(normal) == 0:
    raise RuntimeError("No normal cache entry found.")
if len(seizure) == 0:
    raise RuntimeError("No seizure cache entry found.")

normal_row = normal.iloc[0]
seizure_row = seizure.iloc[0]

def resolve_cache_path(value):
    p = Path(str(value))
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        ROOT / p,
        ROOT / "results" / "stft_cache" / p,
        ROOT / "results" / "stft_cache" / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find cache tensor '{value}'. Tried: "
        + ", ".join(str(c) for c in candidates)
    )

def load_tensor(row):
    path = resolve_cache_path(row[cache_col])
    obj = torch.load(path, map_location="cpu", weights_only=False)

    # Cache may contain a tensor directly or a dictionary.
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        x = None
        for key in ["spectrogram", "stft", "tensor", "data", "x", "input"]:
            if key in obj and isinstance(obj[key], torch.Tensor):
                x = obj[key]
                break
        if x is None:
            tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
            if tensors:
                x = tensors[0]
    else:
        raise TypeError(f"Unsupported cache object type: {type(obj)}")

    if x is None:
        raise ValueError(f"No tensor found in cache file: {path}")

    x = x.float()
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if tuple(x.shape) != (9, 129, 59):
        raise ValueError(f"Expected (9,129,59), got {tuple(x.shape)} from {path}")

    return x.numpy(), path

def make_figure(row, kind, filename):
    x, path = load_tensor(row)

    subject = str(row[subject_col]) if subject_col is not None else "Unknown"
    epoch = str(row[epoch_col]) if epoch_col is not None else "Unknown"

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)

    # Same color scale across all nine channels in this example.
    vmin = float(np.percentile(x, 2))
    vmax = float(np.percentile(x, 98))

    for i, ax in enumerate(axes.flat):
        im = ax.imshow(
            x[i],
            origin="lower",
            aspect="auto",
            extent=[TIMES[0], TIMES[-1], FREQS[0], FREQS[-1]],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(CHANNELS[i])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 128)
        fig.suptitle(
        f"{kind} EEG Time-Frequency Representation\n"
        f"Subject: {subject} | Epoch: {epoch} | Input: 9 × 129 × 59",
        fontsize=15
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Log power")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output = OUT / filename
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}", flush=True)
    print(f"Source cache: {path}", flush=True)
    print(f"Subject: {subject}, Epoch: {epoch}, Label: {row[label_col]}", flush=True)

make_figure(
    normal_row,
    "Normal",
    "09_normal_STFT_example.png"
)

make_figure(
    seizure_row,
    "Seizure",
    "10_seizure_STFT_example.png"
)

print("=" * 60, flush=True)
print("DONE - Graphs 9-10 generated successfully.", flush=True)
