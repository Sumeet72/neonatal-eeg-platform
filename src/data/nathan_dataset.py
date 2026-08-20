from pathlib import Path
import numpy as np
import pandas as pd


class NathanDatasetInfo:
    """
    Loads and validates the Nathan neonatal EEG dataset.

    Dataset structure:

        eeg/
            eeg1.edf
            eeg2.edf
            ...

        lables/
            annotations_2017_A_fixed.csv
            annotations_2017_B.csv
            annotations_2017_C.csv
            clinical_information.csv
            annotations_2017.mat

    The A/B/C annotation files contain:
        rows    = annotation seconds
        columns = subjects

    Final seizure label:
        at least 2 of the 3 annotators mark seizure -> 1
        otherwise -> 0
    """

    def __init__(self, edf_dir, annotations_dir):

        self.edf_dir = Path(edf_dir)
        self.annotations_dir = Path(annotations_dir)

        if not self.edf_dir.exists():
            raise FileNotFoundError(
                f"Nathan EDF directory not found:\n{self.edf_dir}"
            )

        if not self.annotations_dir.exists():
            raise FileNotFoundError(
                f"Nathan annotation directory not found:\n"
                f"{self.annotations_dir}"
            )

        self.annotation_a = None
        self.annotation_b = None
        self.annotation_c = None

        self.subject_labels = {}
        self.edf_map = {}

    def load_annotations(self):
        """
        Load the three annotation CSV files.
        """

        file_a = (
            self.annotations_dir /
            "annotations_2017_A_fixed.csv"
        )

        file_b = (
            self.annotations_dir /
            "annotations_2017_B.csv"
        )

        file_c = (
            self.annotations_dir /
            "annotations_2017_C.csv"
        )

        for file in [file_a, file_b, file_c]:

            if not file.exists():
                raise FileNotFoundError(
                    f"Nathan annotation file not found:\n{file}"
                )

        self.annotation_a = pd.read_csv(file_a)
        self.annotation_b = pd.read_csv(file_b)
        self.annotation_c = pd.read_csv(file_c)

        print(
            "Annotation A:",
            self.annotation_a.shape
        )

        print(
            "Annotation B:",
            self.annotation_b.shape
        )

        print(
            "Annotation C:",
            self.annotation_c.shape
        )

    def validate_annotations(self):
        """
        Make sure A/B/C have identical dimensions
        and subject columns.
        """

        if self.annotation_a is None:
            raise RuntimeError(
                "Call load_annotations() first."
            )

        shapes = {
            self.annotation_a.shape,
            self.annotation_b.shape,
            self.annotation_c.shape,
        }

        if len(shapes) != 1:
            raise ValueError(
                "Nathan annotation files do not have "
                "the same shape."
            )

        columns_a = list(self.annotation_a.columns)
        columns_b = list(self.annotation_b.columns)
        columns_c = list(self.annotation_c.columns)

        if not (
            columns_a == columns_b
            and columns_a == columns_c
        ):
            raise ValueError(
                "Nathan annotation files do not have "
                "identical subject columns."
            )

        print(
            "Annotation validation: PASSED"
        )

        print(
            "Subjects in annotations:",
            len(columns_a)
        )

        print(
            "Annotation time points:",
            self.annotation_a.shape[0]
        )

    def create_majority_labels(self):
        """
        Combine A/B/C using majority voting.

        For every second:

            A + B + C >= 2 -> seizure
            A + B + C <  2 -> non-seizure
        """

        if self.annotation_a is None:
            raise RuntimeError(
                "Load annotations before creating labels."
            )

        self.subject_labels = {}

        for subject_column in self.annotation_a.columns:

            a = pd.to_numeric(
                self.annotation_a[subject_column],
                errors="coerce"
            ).fillna(0).to_numpy()

            b = pd.to_numeric(
                self.annotation_b[subject_column],
                errors="coerce"
            ).fillna(0).to_numpy()

            c = pd.to_numeric(
                self.annotation_c[subject_column],
                errors="coerce"
            ).fillna(0).to_numpy()

            # Majority vote
            labels = (
                (a + b + c) >= 2
            ).astype(np.uint8)

            subject_id = int(subject_column)

            self.subject_labels[subject_id] = labels

        print(
            "Majority-vote labels created:",
            len(self.subject_labels),
            "subjects"
        )

    def find_edf_files(self):
        """
        Find all Nathan EDF files recursively.
        """

        edf_files = list(
            self.edf_dir.rglob("*.edf")
        )

        if not edf_files:
            raise FileNotFoundError(
                f"No EDF files found in:\n"
                f"{self.edf_dir}"
            )

        print(
            "Nathan EDF files found:",
            len(edf_files)
        )

        # Create map:
        #
        # eeg1 -> path
        # eeg2 -> path
        # ...
        #
        self.edf_map = {}

        for file in edf_files:

            stem = file.stem.lower()

            if stem.startswith("eeg"):

                subject_text = stem[3:]

                try:
                    subject_id = int(subject_text)

                    self.edf_map[
                        subject_id
                    ] = str(file)

                except ValueError:
                    pass

        print(
            "EDF subject IDs detected:",
            len(self.edf_map)
        )

    def build_index(self):
        """
        Build a subject-level index connecting:

            subject
              ↓
            EDF
              ↓
        second-by-second labels
        """

        if not self.subject_labels:
            raise RuntimeError(
                "Create majority labels first."
            )

        if not self.edf_map:
            raise RuntimeError(
                "Find EDF files first."
            )

        rows = []

        for subject_id, labels in self.subject_labels.items():

            edf_path = self.edf_map.get(
                subject_id
            )

            if edf_path is None:

                print(
                    f"WARNING: No EDF found for "
                    f"subject {subject_id}"
                )

                continue

            for second, label in enumerate(labels):

                rows.append(
                    {
                        "dataset": "nathan",
                        "subject_id": (
                            f"N{subject_id:03d}"
                        ),
                        "subject_number": subject_id,
                        "edf_path": edf_path,
                        "second": second,
                        "label": int(label),
                    }
                )

        index = pd.DataFrame(rows)

        return index

    def load(self):
        """
        Run the complete Nathan loading pipeline.
        """

        self.load_annotations()

        self.validate_annotations()

        self.create_majority_labels()

        self.find_edf_files()

        index = self.build_index()

        return index


if __name__ == "__main__":

    edf_dir = (
        r"C:\Users\Sumeet Mangat\Downloads\EEG_project"
        r"\dataset\EEG\nathan dataset\eeg"
    )

    annotations_dir = (
        r"C:\Users\Sumeet Mangat\Downloads\EEG_project"
        r"\dataset\EEG\nathan dataset\lables"
    )

    dataset = NathanDatasetInfo(
        edf_dir=edf_dir,
        annotations_dir=annotations_dir
    )

    df = dataset.load()

    print("\n=== NATHAN DATASET SUMMARY ===")

    print(
        "Total second-level records:",
        len(df)
    )

    print(
        "Unique subjects:",
        df["subject_id"].nunique()
    )

    print("\nLabels:")

    print(
        df["label"].value_counts()
    )

    print("\nFirst 10 records:")

    print(
        df.head(10).to_string(
            index=False
        )
    )