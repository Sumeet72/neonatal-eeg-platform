CORRECTED RESNET-18 BASELINE RUNNER

Extract this ZIP into the root of your neonatal-eeg-platform repository so the
file is placed at:
  scripts/run_resnet18_corrected_loso.py

The runner reads:
  results/corrected_epoch_dataset/stft_cache/metadata.csv

It saves only to:
  results/ResNet18_corrected_Nathan_LOSO/

It does not modify the original STFT cache, Transformer results, or final
ResNet-18 checkpoint.

Environment variables:
  RESNET18_COMPARE_METADATA_FILE (optional input metadata path)
  RESNET18_COMPARE_OUTPUT_DIR (optional separate output directory)
  RESNET18_COMPARE_BATCH_SIZE (default 8)
  RESNET18_COMPARE_EPOCHS (default 5)
  RESNET18_COMPARE_PATIENCE (default 2)
  RESNET18_COMPARE_LEARNING_RATE (default 0.0001)
  RESNET18_COMPARE_PRETRAINED (default 1; uses torchvision ImageNet pretrained weights)
  MAX_RESNET18_COMPARE_FOLDS (default 1; set 79 for full Nathan LOSO)

The runner evaluates Nathan subjects only. The training/validation pool contains
the other Nathan subjects plus eligible Sumit normal windows. Thresholds are
selected from validation predictions only, using the 0.10..0.90 grid at 0.05
intervals, with a 0.40 fallback if validation has no positive labels.

This is a ResNet-18 baseline, not a clinically validated model. Use a one-fold
pilot first, review it, and only then start the full LOSO run.
