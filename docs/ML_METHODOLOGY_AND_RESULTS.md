# Neonatal EEG Seizure Detection — ML Methodology and Results

## 1. Project objective

This project investigates deep-learning classification of neonatal EEG epochs as **normal** or **seizure** using time-frequency representations. The system is a research prototype intended to support further investigation; it is **not a clinically validated medical device** and must not be used for diagnosis or treatment decisions.

## 2. Combined dataset

The professor's project requirement is to combine the Nathan/Helsinki and Sumit/O'Toole datasets and prepare a model for use.

| Dataset | Subjects / babies | Epochs | Seizure epochs |
|---|---:|---:|---:|
| Nathan/Helsinki | 79 | 26,735 | 3,396 |
| Sumit/O'Toole | 53 | 169 | 3 |
| **Combined** | **132** | **26,904** | **3,399** |

The combined epoch counts are 23,505 normal and 3,399 seizure. The Sumit labels use the official `metadata.csv` field `seizures_YN` (Y = seizure, N = normal). Sumit/O'Toole contributes very few positive seizure epochs, so its contribution to seizure learning is limited even though it remains in the combined experiment.

## 3. Preprocessing and representation

The verified cached input tensor has shape **(9, 129, 59)**:
- 9 selected EEG channels: F3, F4, C3, C4, O1, O2, T3, T4 and Cz
- 129 frequency bins
- 59 time bins

The EDF pipeline selects/maps channels, filters and resamples signals to 256 Hz, creates 30-second epochs (7,680 samples per channel), z-score normalizes the EEG, and calculates log-power Short-Time Fourier Transform (STFT) features.

STFT configuration:
- FFT size (`n_fft`): 256
- Hop length: 128
- Window length: 256
- Window: Hann
- Centering: disabled (`center=False`)
- Power: squared magnitude of complex STFT
- Log power: `log10(power + 1e-10)`

The already-verified STFT cache contained 26,904 tensors and matched the master epoch index with zero label/cache mismatches. Rebuilding the cache is not required for reviewing the results.

## 4. Models

### Baseline EEG-CNN

The baseline network uses convolutional blocks with 32, 64 and 128 output channels, batch normalization, ReLU activations, max pooling, adaptive average pooling, dropout (0.4), and a two-class linear classifier.

### Final candidate: adapted ResNet-18

ResNet-18 was adapted to accept nine-channel STFT tensors instead of standard three-channel images. The input convolution was changed to nine channels; pretrained three-channel filters were repeated and scaled to initialize the new input convolution. The final classifier maps the 512-dimensional representation to two classes. Class index 0 is normal and class index 1 is seizure.

The adaptation is inspired by transfer learning with ResNet-18; it is not an exact reproduction of any paper's input representation or protocol.

## 5. Training and evaluation

- Dataset: combined Nathan + Sumit
- Total subjects: 132
- Total epochs: 26,904
- Loss: weighted cross-entropy to account for class imbalance
- Optimizer learning rate: 0.0001
- Batch size: 32
- Final all-data training: 5 epochs
- Hardware used for training: NVIDIA GeForce RTX 3050 Laptop GPU
- Evaluation: Leave-One-Subject-Out (LOSO) cross-validation. Each evaluation fold holds out a subject for testing and separates validation subjects from the remaining training subjects.

The threshold reported below is 0.40. It was applied to the pooled LOSO predictions for the model comparison. Threshold selection from pooled held-out test predictions can introduce test-set selection bias, so these figures should be treated as the reported experimental operating point, not as an independently validated clinical threshold.

## 6. Final pooled LOSO results

**ResNet-18 at threshold 0.40**

| Metric | Result |
|---|---:|
| Accuracy | 80.55% |
| Sensitivity / Recall | 61.55% |
| Specificity | 83.29% |
| Precision | 34.76% |
| F1-score | 44.43% |
| ROC-AUC | 0.8153 |
| PR-AUC | 0.4590 |

Confusion matrix values:
- True negatives (TN): 19,578
- False positives (FP): 3,927
- False negatives (FN): 1,307
- True positives (TP): 2,092

The class distribution is imbalanced (23,505 normal vs 3,399 seizure epochs). Accuracy alone is therefore insufficient: an always-normal classifier would obtain about 87.4% accuracy while detecting no seizures. Interpret the sensitivity, precision, F1, ROC-AUC and PR-AUC alongside accuracy.

## 7. Baseline comparison

Both models were evaluated on the same combined dataset under the reported 132-subject LOSO setup and at threshold 0.40.

| Metric | Baseline CNN | ResNet-18 |
|---|---:|---:|
| Accuracy | 76.39% | **80.55%** |
| Sensitivity | 52.49% | **61.55%** |
| Specificity | 79.85% | **83.29%** |
| Precision | 27.36% | **34.76%** |
| F1-score | 35.97% | **44.43%** |
| ROC-AUC | 0.7191 | **0.8153** |
| PR-AUC | 0.2863 | **0.4590** |

ResNet-18 improves every metric shown at the same threshold. However, precision remains limited: many epochs predicted as seizure are false positives. Clinical deployment would require additional validation, stronger false-positive analysis, calibration and expert review.

## 8. Final training run (not the test result)

When trained on the full cached dataset for the deployable checkpoint, the recorded training loss decreased from 0.4397 to 0.1810 and training accuracy increased from 81.20% to 92.59% over five epochs. These are training metrics and must not be presented as held-out evaluation performance. Validation loss/accuracy curves were not saved for this final full-data training run.

## 9. EDF inference smoke test

The trained checkpoint was used to process a real EDF file through the preprocessing, STFT and inference pipeline. The test processed 233 complete 30-second epochs and wrote a CSV of predictions. This confirms end-to-end pipeline execution; it does not constitute independent clinical validation. A small sanity check on two cached epochs also produced the expected labels, but two examples do not establish general performance.

## 10. Limitations and next steps

1. The combined dataset is imbalanced, and Sumit/O'Toole contributes only three seizure-positive epochs.
2. At the reported threshold, seizure precision is 34.76%, indicating a substantial false-positive burden.
3. Performance is reported at epoch level; an operational seizure-event detector also needs event-level metrics (event sensitivity, false alarms per hour, latency, and event merging rules).
4. A model-specific threshold should be selected using validation data only, then evaluated once on held-out subjects.
5. Independent external validation, calibration and assessment by qualified EEG clinicians are required before clinical use.
6. Training curves alone do not rule out overfitting; a saved validation history is needed for that analysis.

## Summary

Under the reported combined-dataset LOSO experiment, adapted ResNet-18 outperformed the baseline EEG-CNN on accuracy, sensitivity, specificity, precision, F1-score, ROC-AUC and PR-AUC at threshold 0.40. The results are promising for an academic research prototype, while false positives, threshold selection, class imbalance and clinical validation remain important limitations.
