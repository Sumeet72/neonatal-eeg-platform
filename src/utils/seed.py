import os
import random

import numpy as np
import torch


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # Make CUDA operations reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(
        f"Random seed set to: {seed}"
    )


if __name__ == "__main__":

    set_seed(42)

    print(
        "Seed test successful."
    )