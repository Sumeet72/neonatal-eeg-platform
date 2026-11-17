"""Make the existing Transformer runner accept alternate metadata/output paths.

Creates a .bak file first and is safe to re-run. This patch also adds an optional
TRANSFORMER_TEST_DATASET filter so the corrected experiment's outer test folds can
be restricted to Nathan subjects while Sumit negative windows remain in training.
"""
from pathlib import Path
import re
import shutil

TARGET = Path("scripts/run_eeg_transformer_loso.py")
BACKUP = Path("scripts/run_eeg_transformer_loso.py.pre_corrected_paths.bak")


def main():
    if not TARGET.exists():
        raise FileNotFoundError(f"Could not find {TARGET}; run from repository root")
    source = TARGET.read_text(encoding="utf-8")
    updated = source

    updated = updated.replace(
        'METADATA_FILE = Path("results/stft_cache/metadata.csv")',
        'METADATA_FILE = Path(os.environ.get("TRANSFORMER_METADATA_FILE", "results/stft_cache/metadata.csv"))',
    )
    updated = updated.replace(
        'OUTPUT_DIR = Path("results/EEGTransformer_LOSO")',
        'OUTPUT_DIR = Path(os.environ.get("TRANSFORMER_OUTPUT_DIR", "results/EEGTransformer_LOSO"))',
    )

    # Filter the outer-test subject list only. All non-held-out subjects, including
    # Sumit negative recordings, remain available to the training/validation pool.
    pattern = re.compile(
        r'(?m)^(?P<indent>[ \t]*)subjects\s*=\s*sorted\(metadata\["subject_id"\]\.astype\(str\)\.unique\(\)\)\s*$'
    )
    def replace_subjects(match):
        indent = match.group("indent")
        block = [
            'all_subjects = sorted(metadata["subject_id"].astype(str).unique())',
            'test_dataset_filter = os.environ.get("TRANSFORMER_TEST_DATASET", "").strip().lower()',
            'if test_dataset_filter:',
            '    test_mask = metadata["dataset"].astype(str).str.lower() == test_dataset_filter',
            '    subjects = sorted(metadata.loc[test_mask, "subject_id"].astype(str).unique())',
            '    if not subjects:',
            '        raise ValueError(',
            '            f"TRANSFORMER_TEST_DATASET={test_dataset_filter!r} matched no subjects in {METADATA_FILE}"',
            '        )',
            '    print(',
            '        f"Outer test subjects restricted to dataset={test_dataset_filter!r}: "',
            '        f"{len(subjects)} subjects; training pool still uses all other datasets."',
            '    )',
            'else:',
            '    subjects = all_subjects',
        ]
        return "\n".join(indent + line for line in block)

    updated, count = pattern.subn(replace_subjects, updated, count=1)
    if count != 1:
        raise RuntimeError(
            "Could not find the expected subjects-list statement in the runner. "
            "No file changes were saved; inspect the current script before patching."
        )
    if updated == source:
        raise RuntimeError("Runner already patched or did not contain the expected path constants; no changes written")

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
    TARGET.write_text(updated, encoding="utf-8")
    print(f"Patched: {TARGET}")
    print(f"Backup:  {BACKUP}")
    print("Runner now supports TRANSFORMER_METADATA_FILE, TRANSFORMER_OUTPUT_DIR, and TRANSFORMER_TEST_DATASET.")
    print("Default values remain unchanged when those environment variables are unset.")


if __name__ == "__main__":
    main()
