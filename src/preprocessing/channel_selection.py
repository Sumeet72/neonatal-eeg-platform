import mne


COMMON_CHANNELS = [
    "F3",
    "F4",
    "C3",
    "C4",
    "O1",
    "O2",
    "T3",
    "T4",
    "Cz"
]


def standardize_channel_names(raw):
    """
    Clean channel names so Nathan and Sumit
    can use the same canonical names.
    """

    rename_map = {}

    for name in raw.ch_names:

        clean_name = name.strip()

        # Remove common EEG prefixes
        clean_name = clean_name.replace(
            "EEG ",
            ""
        )

        clean_name = clean_name.replace(
            "EEG",
            ""
        )

        # Remove reference suffixes
        clean_name = clean_name.replace(
            "-REF",
            ""
        )

        clean_name = clean_name.replace(
            "-Ref",
            ""
        )

        clean_name = clean_name.strip()

        rename_map[name] = clean_name

    raw.rename_channels(rename_map)

    return raw


def select_common_channels(raw):
    """
    Standardize channel names and select the
    common 9 EEG channels in a fixed order.
    """

    raw = standardize_channel_names(raw)

    missing = [
        ch for ch in COMMON_CHANNELS
        if ch not in raw.ch_names
    ]

    if missing:
        raise ValueError(
            f"Missing required channels: {missing}"
        )

    raw.pick(
        COMMON_CHANNELS
    )

    return raw


if __name__ == "__main__":

    print(
        "Common EEG channels:"
    )

    for i, channel in enumerate(
        COMMON_CHANNELS
    ):

        print(
            i,
            "->",
            channel
        )