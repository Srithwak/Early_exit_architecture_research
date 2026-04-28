# Models, Data, and Metrics

## Architectures

| # | Architecture | Channels | Type | Description |
|---|---|---|---|---|
| 1 | Baseline CNN | [64, 64, 64] | Control | Standard 3-stage CNN, no exits |
| 2 | Constant Width | [64, 64, 64] | Existing | Equal channels across stages, with exits |
| 3 | Increasing Width | [32, 64, 128] | Existing | Channels widen with depth |
| 4 | Decreasing Width | [128, 64, 32] | Novel | Funnel — broad early, narrow late |
| 5 | Adaptive Width | [64, 64, 64] + gates | Novel | Learned channel gating (soft pruning) |

### Architecture Details

**Conv1dBlock (Bonn EEG):** stride-2 conv(k=7) → BN → ReLU → MaxPool(2) → conv(k=5) → BN → ReLU → MaxPool(2). Downsamples /16 per stage.

**Conv1dBlockSmall (ECG/MIT-BIH):** stride-2 conv(k=5) → BN → ReLU → conv(k=3) → BN → ReLU → MaxPool(2). Downsamples /4 per stage. Used for short 187-sample sequences.

Each stage has a ClassifierHead (AdaptiveAvgPool → Linear) and a PolicyHead (AdaptiveAvgPool → MLP → Sigmoid) for exit decisions.

Adaptive Width adds a ChannelGate per stage: squeeze → FC → sigmoid → element-wise multiply. Hard-thresholded at 0.5 during inference.

### Parameter Counts

| Architecture | Bonn EEG (params) | ECG/MIT-BIH (params) | Bonn MFLOPs | ECG/MIT-BIH MFLOPs |
|---|---|---|---|---|
| Base CNN / Constant Width | 127,240 | 86,097 | 37.68 | 2.24 |
| Increasing Width | 185,544 | 122,353 | 17.43 | 1.34 |
| Decreasing Width | 192,648 | 128,305 | 112.74 | 6.22 |
| Adaptive Width | 139,720 | 98,577 | 37.68 | 2.24 |

---

## Datasets

| Property | Bonn EEG | ECG Arrhythmia | MIT-BIH |
|---|---|---|---|
| Domain | EEG Seizure Detection | ECG Classification | ECG Classification |
| Train Samples | 280 | 86,798 | 76,612 |
| Sequence Length | 4,097 | 187 | 187 |
| Classes | 2 (binary) | 5 (N, S, V, F, Q) | 5 (N, S, V, F, Q) |
| Input Channels | 6 (1 raw + 5 freq bands) | 6 | 6 |
| Sampling Rate | 173.61 Hz | 125 Hz | 125 Hz |
| Class Imbalance | 75% / 25% | 76.3 / 10.7 / 5.8 / 0.7 / 6.5 | 82.8 / 2.5 / 6.6 / 0.7 / 7.3 |

**Preprocessing:** Z-score normalization + FFT frequency band extraction (delta, theta, alpha, beta, gamma). Each band's power is log-transformed and tiled across the sequence length, producing 6 channels total.

---

## Training Configuration

| Parameter | Bonn EEG | ECG | MIT-BIH |
|---|---|---|---|
| Warmup Epochs | 10 | 1 | 1 |
| Joint Epochs | 10 | 1 | 1 |
| Batch Size | 64 | 128 | 128 |
| Learning Rate | 0.001 | 0.001 | 0.001 |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 |
| Energy Lambda | 0.02 | 0.02 | 0.02 |
| Threshold Strategy | Confidence | Confidence | Confidence |
| Seed | 42 | 42 | 42 |

**Training protocol:** Phase 1 (warmup) trains classifier heads only with lambda=0. Phase 2 (joint) enables exit policy training with energy-aware loss. All models receive the same total epochs for unbiased comparison.

**Loss function:** L = Σ[ p_reach × p_exit × L_CE ] + λ × energy_cost

---

## Results — Bonn EEG (Seizure Detection)

| Model | Accuracy (%) | Recall (%) | F1 (%) | ECE | Energy Red. (%) | Latency (ms) |
|---|---|---|---|---|---|---|
| Baseline CNN | 75.00 | 50.00 | 42.86 | 0.2449 | 0.00 | 2.53 |
| Constant Width | **93.33** | **88.89** | **90.68** | 0.0641 | 29.74 | 2.57 |
| Increasing Width ⭐ | **93.33** | **88.89** | **90.68** | 0.1086 | **42.11** | **1.46** |
| Decreasing Width | **93.33** | **88.89** | **90.68** | **0.0257** | 15.83 | 9.80 |
| Adaptive Width | 75.00 | 50.00 | 42.86 | 0.2434 | 48.18* | 5.39 |

*Adaptive Width collapsed to majority-class prediction (gate collapse on 280 samples).

---

## Results — ECG Arrhythmia (5-Class)

| Model | Accuracy (%) | Recall (%) | F1 (%) | ECE | Energy Red. (%) | Latency (ms) |
|---|---|---|---|---|---|---|
| **Baseline CNN** | **93.41** | **92.51** | **83.32** | **0.0125** | 0.00 | 0.43 |
| Constant Width | 88.77 | 84.42 | 79.79 | 0.1796 | 7.81 | 0.51 |
| Increasing Width ⭐ | 88.94 | 87.52 | 79.54 | 0.1452 | **25.69** | **0.25** |
| Decreasing Width | 88.05 | 89.18 | 76.51 | 0.1638 | 2.95 | 0.67 |
| Adaptive Width | 88.68 | 81.72 | 76.66 | 0.0924 | 17.11 | 0.36 |

*Note: Trained only 2 total epochs (1+1). Exit branches under-trained.*

---

## Results — MIT-BIH Arrhythmia (5-Class)

| Model | Accuracy (%) | Recall (%) | F1 (%) | ECE | Energy Red. (%) | Latency (ms) |
|---|---|---|---|---|---|---|
| **Baseline CNN** | **94.03** | **89.90** | **82.11** | **0.0125** | 0.00 | 0.43 |
| Constant Width | 91.07 | 69.94 | 66.60 | 0.2581 | 8.17 | 0.54 |
| Increasing Width ⭐ | 89.78 | 77.95 | 72.07 | 0.2445 | **33.60** | **0.31** |
| Decreasing Width | 92.06 | 73.92 | 71.36 | 0.2529 | 2.27 | 0.67 |
| Adaptive Width | 89.11 | 85.99 | 73.00 | 0.1851 | 14.14 | 0.34 |

*Note: Trained only 2 total epochs (1+1). Exit branches under-trained.*

---

## Cross-Dataset: Energy Reduction (%)

| Model | Bonn EEG | ECG | MIT-BIH |
|---|---|---|---|
| Constant Width | 29.74 | 7.81 | 8.17 |
| **Increasing Width** ⭐ | **42.11** | **25.69** | **33.60** |
| Decreasing Width | 15.83 | 2.95 | 2.27 |
| Adaptive Width | 48.18* | 17.11 | 14.14 |

Increasing Width `[32, 64, 128]` is Pareto-optimal across all datasets.

---

## Cross-Dataset: Calibration (ECE)

| Model | Bonn EEG | ECG | MIT-BIH |
|---|---|---|---|
| Baseline CNN | 0.2449 | **0.0125** | **0.0125** |
| Constant Width | 0.0641 | 0.1796 | 0.2581 |
| Increasing Width | 0.1086 | 0.1452 | 0.2445 |
| Decreasing Width | **0.0257** | 0.1638 | 0.2529 |
| Adaptive Width | 0.2434 | 0.0924 | 0.1851 |

---

## Threshold Strategy Comparison (Bonn EEG)

| Strategy | Accuracy (%) | F1 (%) | Energy Red. (%) |
|---|---|---|---|
| **Confidence** | **93.3** | **90.7** | **28.4** |
| Entropy | 86.7 | 79.2 | 20.1 |
| Patience | 91.7 | 88.1 | 3.2 |

Confidence-based thresholding wins across all metrics.

---

## Structured Pruning (Bonn EEG)

| Prune Ratio | Accuracy (%) | FLOPs Reduction (%) |
|---|---|---|
| 0% | 93.3 | 0 |
| 10% | 81.7 | 19 |
| 25% | 83.3 | 43 |
| 50% | 80.0 | 70 |

Steep accuracy cliff — early exits are a more graceful efficiency mechanism than pruning.

---

## Model Size Scaling (Bonn EEG)

| Size | Channels | Params | Accuracy (%) | Energy Red. (%) |
|---|---|---|---|---|
| Tiny | [16, 16, 16] | ~8K | ~80 | ~30 |
| Small | [32, 32, 32] | ~32K | ~90 | ~35 |
| Medium | [64, 64, 64] | ~127K | ~93 | ~30 |
| Large | [128, 128, 128] | ~500K | ~88 | ~25 |
| XLarge | [256, 256, 256] | ~2M | ~77 | ~20 |

Sweet spot is Small–Medium (32–64 channels). XLarge overfits on the small Bonn dataset.

---

## Key Findings

1. **Early exits save 25–42% energy** across all datasets (Increasing Width `[32→64→128]`).
2. **Early exits regularize small datasets** — F1 jumps 42.86% → 90.68% on Bonn (280 samples).
3. **Training budget matters** — 2 epochs is insufficient for ECG/MIT-BIH exit branches.
4. **Decreasing Width** has best calibration (ECE=0.026) but worst efficiency.
5. **Adaptive Width** collapses on small data; needs ≥50K samples.
6. **Confidence thresholding** outperforms entropy and patience strategies.
7. **Calibration degrades** with early exits on large datasets (ECE 0.14–0.26).
