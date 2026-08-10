import sys
import torch
import numpy as np
import scipy
import pandas as pd
import mne
import sklearn
import yaml
import tqdm
import matplotlib


print("=" * 50)
print("NEONATAL EEG PROJECT - ENVIRONMENT CHECK")
print("=" * 50)

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"SciPy        : {scipy.__version__}")
print(f"Pandas       : {pd.__version__}")
print(f"MNE          : {mne.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"PyYAML       : {yaml.__version__}")
print(f"tqdm         : {tqdm.__version__}")
print(f"Matplotlib   : {matplotlib.__version__}")

print("\nGPU CHECK")
print("-" * 50)

print(f"CUDA available : {torch.cuda.is_available()}")
print(f"CUDA version   : {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"GPU count      : {torch.cuda.device_count()}")

    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print(f"GPU test       : PASSED")
    print(f"Tensor device  : {y.device}")
else:
    print("GPU test       : FAILED")

print("=" * 50)