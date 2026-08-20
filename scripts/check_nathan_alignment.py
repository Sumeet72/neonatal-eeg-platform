from pathlib import Path
import yaml
import mne


with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)


edf_dir = Path(config["nathan"]["edf_dir"])


edf_files = sorted(
    edf_dir.rglob("*.edf"),
    key=lambda x: int(x.stem[3:]) if x.stem.lower().startswith("eeg") else 9999
)


print("Nathan EDF alignment check")
print("=" * 50)

for edf_file in edf_files:

    raw = mne.io.read_raw_edf(
        edf_file,
        preload=False,
        verbose=False
    )

    duration = raw.times[-1]

    subject_id = edf_file.stem

    print(
        f"{subject_id}: "
        f"{duration:.2f} seconds | "
        f"{raw.info['sfreq']} Hz | "
        f"{len(raw.ch_names)} channels"
    )