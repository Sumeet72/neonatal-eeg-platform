from pathlib import Path
import pandas as pd


class SumitDatasetInfo:
    """
    Loads and validates the Sumit neonatal EEG metadata
    and connects metadata records to their EDF files.
    """

    def __init__(self, edf_dir, labels_dir):
        self.edf_dir = Path(edf_dir)
        self.labels_dir = Path(labels_dir)

        if not self.edf_dir.exists():
            raise FileNotFoundError(
                f"Sumit EDF directory not found:\n{self.edf_dir}"
            )

        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f"Sumit labels directory not found:\n{self.labels_dir}"
            )

        self.metadata = None
        self.grades = None
        self.edf_map = {}

    def load_labels(self):
        """Load metadata.csv and eeg_grades.csv."""

        metadata_file = self.labels_dir / "metadata.csv"
        grades_file = self.labels_dir / "eeg_grades.csv"

        if not metadata_file.exists():
            raise FileNotFoundError(
                f"metadata.csv not found:\n{metadata_file}"
            )

        if not grades_file.exists():
            raise FileNotFoundError(
                f"eeg_grades.csv not found:\n{grades_file}"
            )

        self.metadata = pd.read_csv(metadata_file)
        self.grades = pd.read_csv(grades_file)

        print("Sumit metadata loaded:", self.metadata.shape)
        print("Sumit grades loaded:", self.grades.shape)

    def create_seizure_label(self):
        """
        Convert seizures_YN into a numeric binary label.

        Y = 1 = seizure
        N = 0 = non-seizure
        """

        if self.metadata is None:
            raise RuntimeError(
                "Call load_labels() before create_seizure_label()."
            )

        if "seizures_YN" not in self.metadata.columns:
            raise ValueError(
                "Column 'seizures_YN' was not found in metadata.csv."
            )

        self.metadata["label"] = (
            self.metadata["seizures_YN"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({
                "Y": 1,
                "N": 0
            })
        )

        if self.metadata["label"].isna().any():
            unknown = (
                self.metadata.loc[
                    self.metadata["label"].isna(),
                    "seizures_YN"
                ]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                f"Unknown values found in seizures_YN: {unknown}"
            )

    def find_edf_files(self):
        """Find all Sumit EDF files recursively."""

        edf_files = list(self.edf_dir.rglob("*.edf"))

        if not edf_files:
            raise FileNotFoundError(
                f"No EDF files found in:\n{self.edf_dir}"
            )

        self.edf_map = {
            file.stem: str(file)
            for file in edf_files
        }

        print("Sumit EDF files found:", len(self.edf_map))

    def build_index(self):
        """
        Match each metadata record with its EDF file.
        """

        if self.metadata is None:
            raise RuntimeError(
                "Load metadata before building the index."
            )

        if not self.edf_map:
            raise RuntimeError(
                "Find EDF files before building the index."
            )

        self.metadata["edf_path"] = (
            self.metadata["file_ID"]
            .astype(str)
            .map(self.edf_map)
        )

        missing = self.metadata["edf_path"].isna()

        if missing.any():
            missing_ids = (
                self.metadata.loc[missing, "file_ID"]
                .astype(str)
                .tolist()
            )

            print(
                f"WARNING: {len(missing_ids)} metadata records "
                f"do not have matching EDF files."
            )

            print("First missing IDs:")
            print(missing_ids[:10])

        return self.metadata

    def load(self):
        """Run the complete Sumit loading pipeline."""

        self.load_labels()
        self.create_seizure_label()
        self.find_edf_files()
        df = self.build_index()

        return df


if __name__ == "__main__":

    # Temporary direct test.
    # The final training pipeline will obtain these paths
    # from configs/paths.yaml.

    edf_dir = (
        r"C:\Users\Sumeet Mangat\Downloads\EEG_project"
        r"\dataset\EEG\sumit dataset\data"
    )

    labels_dir = (
        r"C:\Users\Sumeet Mangat\Downloads\EEG_project"
        r"\dataset\EEG\sumit dataset\labels"
    )

    dataset = SumitDatasetInfo(
        edf_dir=edf_dir,
        labels_dir=labels_dir
    )

    df = dataset.load()

    print("\n=== SUMIT DATASET SUMMARY ===")

    print("Total metadata records:", len(df))

    print("\nSeizure labels:")
    print(df["label"].value_counts())

    print("\nUnique babies:")
    print(df["baby_ID"].nunique())

    print("\nSampling frequencies:")
    print(df["sampling_freq"].value_counts())

    print("\nFirst 5 records:")
    print(
        df[
            [
                "file_ID",
                "baby_ID",
                "epoch_number",
                "grade",
                "sampling_freq",
                "seizures_YN",
                "label",
                "edf_path"
            ]
        ].head()
    )