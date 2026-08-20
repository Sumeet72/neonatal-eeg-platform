import torch


def get_device():
    """
    Select CUDA GPU if available, otherwise CPU.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print(
            "Using GPU:",
            torch.cuda.get_device_name(0)
        )

    else:
        device = torch.device("cpu")

        print("CUDA not available. Using CPU.")

    return device


def print_device_info():

    print(
        "PyTorch version:",
        torch.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "CUDA version:",
            torch.version.cuda
        )

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        properties = (
            torch.cuda.get_device_properties(0)
        )

        print(
            "GPU memory:",
            round(
                properties.total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )


if __name__ == "__main__":

    print_device_info()

    device = get_device()

    print(
        "Selected device:",
        device
    )