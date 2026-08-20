import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut


def create_loso_splits(
    dataframe
):
    """
    Create Leave-One-Subject-Out splits.

    Every epoch belonging to the same subject
    stays in the same fold.
    """

    df = dataframe.reset_index(
        drop=True
    )

    groups = df["subject_id"]

    logo = LeaveOneGroupOut()

    splits = []

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        logo.split(
            df,
            df["label"],
            groups=groups
        ),
        start=1
    ):

        train_df = df.iloc[
            train_idx
        ].reset_index(drop=True)

        test_df = df.iloc[
            test_idx
        ].reset_index(drop=True)

        train_subjects = set(
            train_df["subject_id"]
        )

        test_subjects = set(
            test_df["subject_id"]
        )

        # ---------------------------------
        # Leakage check
        # ---------------------------------

        overlap = (
            train_subjects
            & test_subjects
        )

        if overlap:

            raise RuntimeError(
                "Subject leakage detected: "
                f"{overlap}"
            )

        splits.append(
            {
                "fold": fold,
                "train": train_df,
                "test": test_df,
                "test_subjects": test_subjects
            }
        )

    return splits


if __name__ == "__main__":

    master_path = (
        "results/master_epochs.csv"
    )

    df = pd.read_csv(
        master_path
    )

    print(
        "Total epochs:",
        len(df)
    )

    print(
        "Unique subjects:",
        df["subject_id"].nunique()
    )

    splits = create_loso_splits(
        df
    )

    print(
        "\nTotal LOSO folds:",
        len(splits)
    )

    # -------------------------------------
    # Inspect first three folds
    # -------------------------------------

    for split in splits[:3]:

        train_df = split["train"]
        test_df = split["test"]

        print(
            "\nFold:",
            split["fold"]
        )

        print(
            "Test subject:",
            sorted(
                split["test_subjects"]
            )
        )

        print(
            "Training epochs:",
            len(train_df)
        )

        print(
            "Testing epochs:",
            len(test_df)
        )

        print(
            "Train subjects:",
            train_df[
                "subject_id"
            ].nunique()
        )

        print(
            "Test subjects:",
            test_df[
                "subject_id"
            ].nunique()
        )

        overlap = (
            set(train_df["subject_id"])
            &
            set(test_df["subject_id"])
        )

        print(
            "Subject overlap:",
            overlap
        )

    print(
        "\nLOSO split test successful."
    )