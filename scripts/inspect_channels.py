from pathlib import Path
import yaml
import mne


with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)


def inspect_dataset(name, edf_dir):

    edf_dir = Path(edf_dir)

    edf_files = sorted(
        edf_dir.rglob("*.edf")
    )

    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    print("EDF files:", len(edf_files))

    for edf_file in edf_files[:5]:

        raw = mne.io.read_raw_edf(
            edf_file,
            preload=False,
            verbose=False
        )

        print("\nFile:", edf_file.name)

        print(
            "Sampling frequency:",
            raw.info["sfreq"]
        )

        print(
            "Number of channels:",
            len(raw.ch_names)
        )

        print(
            "Channels:"
        )

        print(
            raw.ch_names
        )


inspect_dataset(
    "Nathan",
    config["nathan"]["edf_dir"]
)


inspect_dataset(
    "Sumit",
    config["sumit"]["edf_dir"]
)