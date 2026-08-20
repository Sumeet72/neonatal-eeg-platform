from pathlib import Path
import yaml
import pandas as pd


with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)


edf_dir = Path(config["sumit"]["edf_dir"])
labels_dir = Path(config["sumit"]["labels_dir"])


metadata_file = labels_dir / "metadata.csv"

metadata = pd.read_csv(metadata_file)


print("Sumit metadata:", metadata.shape)


# Find all EDF files
edf_files = list(edf_dir.rglob("*.edf"))

print("Sumit EDF files found:", len(edf_files))


# Map EDF filename/stem to full path
edf_map = {
    file.stem: str(file)
    for file in edf_files
}


# Match every metadata record with its EDF
metadata["edf_path"] = (
    metadata["file_ID"]
    .astype(str)
    .map(edf_map)
)


# Convert seizure label
metadata["label"] = (
    metadata["seizures_YN"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({
        "Y": 1,
        "N": 0
    })
)


# Check for missing EDF matches
missing_edf = metadata["edf_path"].isna()

if missing_edf.any():

    print(
        "\nWARNING:",
        missing_edf.sum(),
        "metadata records have no matching EDF."
    )

    print(
        metadata.loc[
            missing_edf,
            "file_ID"
        ].tolist()
    )

else:

    print(
        "EDF matching: PASSED"
    )


# Create final epoch-level table
df = metadata[
    [
        "file_ID",
        "baby_ID",
        "epoch_number",
        "grade",
        "sampling_freq",
        "reference",
        "seizures_YN",
        "label",
        "edf_path"
    ]
].copy()


df.insert(
    0,
    "dataset",
    "sumit"
)


# Save result
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)


output_file = output_dir / "sumit_epochs.csv"

df.to_csv(
    output_file,
    index=False
)


print("\n=== SUMIT EPOCH SUMMARY ===")

print(
    "Total epochs:",
    len(df)
)

print(
    "Unique babies:",
    df["baby_ID"].nunique()
)

print("\nLabels:")

print(
    df["label"].value_counts()
)

print("\nSampling frequencies:")

print(
    df["sampling_freq"].value_counts()
)

print("\nFirst 5 records:")

print(
    df.head().to_string(index=False)
)

print("\nSaved to:")

print(output_file)