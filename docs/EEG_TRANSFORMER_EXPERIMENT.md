# Pure EEG Spatio-Temporal Transformer Experiment

## Purpose

This experiment adds a **pure Transformer** classifier to the existing neonatal EEG project. It does not use CNN or convolutional layers. The existing ResNet-18 and baseline CNN remain unchanged.

## Architecture

File: `src/models/eeg_transformer.py`

1. Input is the existing log-power STFT tensor shaped `[B, 9, 129, 59]`.
2. Each channel/time frame's 129-bin spectrum is normalized and linearly projected to a 64-dimensional token.
3. Learnable temporal positional embeddings and channel identity embeddings are added.
4. A two-layer temporal Transformer encoder models dependencies across the 59 frames independently for each channel.
5. A one-layer channel Transformer encoder models relationships across the 9 channels at each time frame.
6. Mean pooling and a small MLP output two logits: class 0 (normal) and class 1 (seizure).

This factorized design applies self-attention in time and across channels. Frequency bins are supplied as the spectral feature vector to a linear projection; the network does not run a separate attention stack over frequency bins.

### Initial hyperparameters

| Parameter | Initial value |
|---|---:|
| Embedding dimension | 64 |
| Attention heads | 4 |
| Temporal encoder layers | 2 |
| Channel encoder layers | 1 |
| Feed-forward dimension | 256 |
| Dropout | 0.20 |
| Batch size | 8 |
| Maximum training epochs per fold | 5 |
| Learning rate | 0.0001 |
| Loss | Inverse-frequency weighted cross-entropy |
| Optimizer | AdamW |

These are starting settings, not tuned or validated optimums. Batch size, layers, and epochs can be overridden using the environment variables below.

## First test: model forward and backward pass

Run these commands from the repository root after activating your existing environment:

```powershell
python -m src.models.eeg_transformer
```

The smoke test uses a synthetic tensor only to check the model's expected input/output shape and backward pass. It does not evaluate seizure performance or represent EEG data.

Expected output includes:
- Input shape: `(2, 9, 129, 59)`
- Logit shape: `(2, 2)`
- Probability shape: `(2, 2)`
- Finite loss
- Backward pass: success

## First training run: one outer fold

This uses the existing `results/stft_cache/metadata.csv` and cached `.pt` tensors. It does not rebuild the cache.

```powershell
$env:MAX_LOSO_FOLDS = "1"
python -m scripts.run_eeg_transformer_loso
```

The one-fold run is a pipeline smoke/pilot run, not a final performance result. The script saves per-subject prediction CSVs, training history, fold metrics, and a current pooled summary under:

`results/EEGTransformer_LOSO/`

If memory is tight, set a smaller batch size before running:

```powershell
$env:TRANSFORMER_BATCH_SIZE = "4"
```

Other optional environment settings:

```powershell
$env:TRANSFORMER_EPOCHS = "5"
$env:TRANSFORMER_PATIENCE = "2"
$env:TRANSFORMER_LEARNING_RATE = "0.0001"
```

## Full 132-subject LOSO run

After the one-fold smoke test runs successfully, run or resume the full outer LOSO experiment:

```powershell
$env:MAX_LOSO_FOLDS = "132"
python -m scripts.run_eeg_transformer_loso
```

The runner resumes completed subject folds using the saved fold summary and prediction files. Do not delete that output directory if you intend to resume.

Each outer fold holds out one complete subject for testing. A separate subject-group split is made from the remaining subjects to obtain training and validation sets. There are assertions that training, validation, and held-out subject groups do not overlap.

For each outer fold, the operating threshold is selected from the validation predictions only using a 0.10–0.90 grid and F1. The selected threshold is then applied to the held-out subject. If the validation subset has no seizure-positive epochs, the script records a 0.40 fallback and a reason. Pooled ROC-AUC and PR-AUC are calculated from held-out probabilities; threshold metrics use each outer fold's validation-selected threshold.

## Output files

- `results/EEGTransformer_LOSO/predictions/<subject>.csv`: held-out epoch probabilities and predictions
- `results/EEGTransformer_LOSO/training_history/<subject>.csv`: training/validation curves for the fold
- `results/EEGTransformer_LOSO/fold_metrics.csv`: per-fold metrics and chosen validation thresholds
- `results/EEGTransformer_LOSO/pooled_metrics.json`: pooled results and whether all subjects have been evaluated

A summary marked `partial_loso_pilot` must not be reported as the completed LOSO result.

## Important data-labeling caveat

The current cache contains labels propagated from the existing master index. The Sumit/O'Toole metadata's seizure label is recording-level. The existing master-building logic chooses a 30-second interval from an `epoch_number` field; that mapping needs to be confirmed against the actual recording format and seizure annotation timing before those segments can be treated as confidently labeled 30-second seizure/normal examples.

Cache/master label consistency confirms that values agree between files, but does not prove temporal label validity. This script prints and records the caveat. Resolve the label/segment alignment before interpreting final scientific performance.

## Model comparison caveat

Keep the ResNet-18 result as the existing reference and do not overwrite its checkpoint or metrics. This Transformer's threshold metrics use validation-only threshold selection. For a strictly comparable thesis table of threshold-dependent metrics, re-evaluate the ResNet-18 baseline with the same held-out subject splits and validation-only threshold selection. ROC-AUC and PR-AUC are threshold-independent, though valid ground-truth labels remain essential.

## Limitations

- This model predicts seizure versus normal at epoch level.
- It does not automatically provide validated seizure onset/offset precision below the epoch resolution.
- It does not provide validated channel-level seizure source localization.
- It does not make clinical diagnoses.
- All results are research outputs requiring independent validation and expert review.
