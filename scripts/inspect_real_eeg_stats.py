from pathlib import Path
import yaml
import mne
import numpy as np

from src.preprocessing.channel_selection import select_common_channels


with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)


nathan_file = sorted(
    Path(config["nathan"]["edf_dir"]).rglob("*.edf")
)[0]

print("=" * 60)
print("NATHAN EEG CHANNEL STATISTICS")
print("=" * 60)

raw = mne.io.read_raw_edf(
    nathan_file,
    preload=True,
    verbose=False
)

raw = select_common_channels(raw)

data = raw.get_data()

for i, channel in enumerate(raw.ch_names):

    signal = data[i]

    print(
        f"{i}: {channel:>3} | "
        f"min={signal.min():.8e} | "
        f"max={signal.max():.8e} | "
        f"mean={signal.mean():.8e} | "
        f"std={signal.std():.8e}"
    )