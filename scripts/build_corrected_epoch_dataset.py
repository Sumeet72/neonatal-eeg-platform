"""Build a corrected, separate epoch manifest without touching historical data.

- Keeps the existing Nathan annotation-derived epochs unchanged.
- Expands each Sumit recording labeled seizures_YN == 'N' into 30-second windows.
- Includes seizure-positive Sumit recordings with unknown window labels (-1).
  Their recording-level positive labels will be used by multiple-instance learning.
- Writes only under results/corrected_epoch_dataset/.
"""
from pathlib import Path
import json
import math

import mne
import pandas as pd

SOURCE_MASTER = Path("results/master_epochs.csv")
SUMIT_TABLE = Path("results/sumit_epochs.csv")
OUTPUT_DIR = Path("results/corrected_epoch_dataset")
OUTPUT_MASTER = OUTPUT_DIR / "master_epochs.csv"
AUDIT_FILE = OUTPUT_DIR / "sumit_recording_audit.csv"
WINDOW_SECONDS = 30.0

REQUIRED_MASTER_COLUMNS = {
    "epoch_id", "dataset", "subject_id", "edf_path",
    "start_second", "end_second", "label"
}


def main():
    if not SOURCE_MASTER.exists():
        raise FileNotFoundError(f"Missing existing master manifest: {SOURCE_MASTER}")
    if not SUMIT_TABLE.exists():
        raise FileNotFoundError(f"Missing Sumit table: {SUMIT_TABLE}")

    old_master = pd.read_csv(SOURCE_MASTER)
    missing = REQUIRED_MASTER_COLUMNS - set(old_master.columns)
    if missing:
        raise ValueError(f"{SOURCE_MASTER} is missing columns: {sorted(missing)}")

    dataset_names = old_master["dataset"].astype(str).str.lower()
    nathan = old_master.loc[dataset_names == "nathan", list(REQUIRED_MASTER_COLUMNS)].copy()
    if nathan.empty:
        raise ValueError("No existing Nathan rows found in results/master_epochs.csv")
    nathan["dataset"] = "nathan"
    nathan["subject_id"] = nathan["subject_id"].astype(str)
    nathan["label"] = pd.to_numeric(nathan["label"], errors="raise").astype(int)
    nathan["epoch_id"] = pd.to_numeric(nathan["epoch_id"], errors="raise").astype(int)

    sumit = pd.read_csv(SUMIT_TABLE)
    required_sumit = {"file_ID", "baby_ID", "label", "edf_path", "seizures_YN"}
    missing_sumit = required_sumit - set(sumit.columns)
    if missing_sumit:
        raise ValueError(f"{SUMIT_TABLE} is missing columns: {sorted(missing_sumit)}")

    sumit["label"] = pd.to_numeric(sumit["label"], errors="raise").astype(int)
    if not set(sumit["label"].unique()).issubset({0, 1}):
        raise ValueError("Sumit labels must be binary 0/1")

    # Confirm the numeric label agrees with the source recording-level Y/N label.
    yn = sumit["seizures_YN"].astype(str).str.strip().str.upper().map({"Y": 1, "N": 0})
    if yn.isna().any() or not (yn.to_numpy() == sumit["label"].to_numpy()).all():
        raise ValueError("Sumit label and seizures_YN disagree or contain unknown values")
    if sumit["edf_path"].isna().any():
        raise ValueError("Sumit has missing EDF paths; fix the source mapping first")
    if sumit["edf_path"].astype(str).duplicated().any():
        raise ValueError("Sumit table contains duplicate EDF paths; refusing ambiguous expansion")

    positive_recordings = sumit.loc[sumit["label"] == 1].copy()
    negative_recordings = sumit.loc[sumit["label"] == 0].copy()

    next_epoch_id = int(pd.to_numeric(old_master["epoch_id"], errors="raise").max()) + 1
    sumit_windows = []
    audit_rows = []
    bad_recordings = []

    for row in sumit.itertuples(index=False):
        edf_path = Path(str(row.edf_path))
        record_id = str(row.file_ID)
        if not edf_path.exists():
            bad_recordings.append({"file_ID": record_id, "edf_path": str(edf_path), "error": "EDF file not found"})
            continue

        raw = None
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
            duration = float(raw.n_times / raw.info["sfreq"])
        except Exception as exc:
            bad_recordings.append({"file_ID": record_id, "edf_path": str(edf_path), "error": repr(exc)})
            continue
        finally:
            if raw is not None:
                raw.close()

        full_windows = int(math.floor((duration + 1e-6) / WINDOW_SECONDS))
        remainder = max(0.0, duration - full_windows * WINDOW_SECONDS)
        audit_rows.append({
            "file_ID": record_id,
            "baby_ID": str(row.baby_ID),
            "edf_path": str(edf_path),
            "recording_label": int(row.label),
            "duration_seconds": duration,
            "full_30s_windows": full_windows,
            "unused_tail_seconds": remainder,
            "window_label_status": (
                "known_normal" if int(row.label) == 0
                else "unknown_instance_labels_recording_positive"
            ),
        })

        for window_index in range(full_windows):
            start = float(window_index * WINDOW_SECONDS)
            end = float(start + WINDOW_SECONDS)
            sumit_windows.append({
                "epoch_id": next_epoch_id,
                "dataset": "sumit",
                "subject_id": str(row.baby_ID),
                "edf_path": str(edf_path),
                "start_second": start,
                "end_second": end,
                "label": 0 if int(row.label) == 0 else -1,
            })
            next_epoch_id += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(AUDIT_FILE, index=False)

    if bad_recordings:
        bad_path = OUTPUT_DIR / "sumit_recording_errors.csv"
        pd.DataFrame(bad_recordings).to_csv(bad_path, index=False)
        raise RuntimeError(
            f"Could not validate {len(bad_recordings)} Sumit recordings. "
            f"See {bad_path}; no corrected master manifest was written."
        )

    sumit_master = pd.DataFrame(sumit_windows, columns=[
        "epoch_id", "dataset", "subject_id", "edf_path",
        "start_second", "end_second", "label"
    ])
    master = pd.concat([nathan, sumit_master], ignore_index=True)
    master = master[[
        "epoch_id", "dataset", "subject_id", "edf_path",
        "start_second", "end_second", "label"
    ]].sort_values("epoch_id").reset_index(drop=True)

    if master["epoch_id"].duplicated().any():
        raise RuntimeError("Corrected manifest has duplicate epoch_id values")
    if master[["start_second", "end_second", "label"]].isna().any().any():
        raise RuntimeError("Corrected manifest has missing window metadata")
    if not (master["end_second"] > master["start_second"]).all():
        raise RuntimeError("Invalid window range found")

    master.to_csv(OUTPUT_MASTER, index=False)
    summary = {
        "manifest": str(OUTPUT_MASTER),
        "nathan_rows_preserved": int(len(nathan)),
        "sumit_negative_recordings": int(len(negative_recordings)),
        "sumit_positive_recordings_included_with_weak_labels": int(len(positive_recordings)),
        "sumit_30s_windows_added": int(len(sumit_master)),
        "total_rows": int(len(master)),
        "label_counts": {str(k): int(v) for k, v in master["label"].value_counts().sort_index().items()},
        "unique_subjects": int(master["subject_id"].nunique()),
        "window_seconds_for_sumit": WINDOW_SECONDS,
        "note": (
            "All Sumit recordings are represented. Negative recordings contribute normal window labels; "
            "positive recordings contribute label=-1 because their instance labels are unknown. "
            "Use recording-level multiple-instance learning for the positive recordings. "
            "Evaluate on held-out Nathan subjects only."
        ),
    }
    (OUTPUT_DIR / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("CORRECTED MANIFEST BUILT (historical files untouched)")
    print(f"Nathan epochs preserved: {len(nathan):,}")
    print(f"Sumit negative recordings expanded: {len(negative_recordings):,}")
    print(f"Sumit positive recordings included with weak labels: {len(positive_recordings):,}")
    print(f"New Sumit 30-second windows: {len(sumit_master):,}")
    print(f"Total windows: {len(master):,}")
    print("Label counts:")
    print(master["label"].value_counts().sort_index().to_string())
    print(f"Saved: {OUTPUT_MASTER}")
    print(f"Recording duration audit: {AUDIT_FILE}")


if __name__ == "__main__":
    main()
