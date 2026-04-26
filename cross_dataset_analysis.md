# Cross-Dataset Analysis: Early-Exit Architecture Research

> **Generated**: April 25, 2026  
> **Threshold Strategy**: Confidence-based | **Energy λ**: 0.02 | **Trials**: 1 per configuration

---

## 1. Datasets Overview

| Property | Bonn EEG | ECG Arrhythmia | MIT-BIH Arrhythmia |
|---|---|---|---|
| **Domain** | EEG Seizure Detection | ECG Classification | ECG Classification |
| **Train Samples** | 280 | 86,798 | 76,612 |
| **Sequence Length** | 4,097 | 187 | 187 |
| **Num Classes** | 2 (binary) | 5 (multi-class) | 5 (multi-class) |
| **Input Channels** | 6 (1 raw + 5 freq bands) | 6 (1 raw + 5 freq bands) | 6 (1 raw + 5 freq bands) |
| **Class Imbalance** | 75% / 25% | 76.3% / 10.7% / 5.8% / 0.7% / 6.5% | 82.8% / 2.5% / 6.6% / 0.7% / 7.3% |
| **Training Epochs** | 10 warmup + 10 joint | 1 warmup + 1 joint | 1 warmup + 1 joint |
| **Batch Size** | 64 | 128 | 128 |

> **NOTE:** The Bonn dataset is extremely small (280 training samples) with long sequences (4,097 timesteps), while the ECG and MIT-BIH datasets are large-scale (76K–87K samples) with short sequences (187 timesteps). The ECG/MIT-BIH models were trained for only 2 total epochs vs. 20 for Bonn, which significantly impacts convergence.

---

## 2. Per-Dataset Results

### 2.1 Bonn EEG (Seizure Detection — Binary)

| Model | Params | MFLOPs | Accuracy (%) | Recall (%) | F1 (%) | ECE | Energy Red. (%) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **Base CNN (Control)** | 127,240 | 37.68 | 75.00 | 50.00 | 42.86 | 0.2449 | 0.00 | 2.53 |
| **Constant Width** | 127,240 | 37.68 | **93.33** | **88.89** | **90.68** | 0.0641 | 29.74 | 2.57 |
| **Increasing Width** | 185,544 | 17.43 | **93.33** | **88.89** | **90.68** | 0.1086 | **42.11** | **1.46** |
| **Decreasing Width** | 192,648 | 112.74 | **93.33** | **88.89** | **90.68** | **0.0257** | 15.83 | 9.80 |
| **Adaptive Width** | 139,720 | 37.68 | 75.00 | 50.00 | 42.86 | 0.2434 | 48.18 | 5.39 |

**Key Findings (Bonn)**:
- All early-exit architectures (Constant, Increasing, Decreasing) **massively outperform** the Base CNN, jumping from 42.86% → 90.68% F1. The baseline predicts the majority class almost exclusively.
- **Increasing Width** achieves the best accuracy-efficiency tradeoff: same 93.33% accuracy as others but with **42.11% energy reduction** and lowest latency (1.46 ms).
- **Decreasing Width** has the best calibration (ECE = 0.0257) but at the cost of 6.5x higher MFLOPs (112.74 vs 17.43).
- **Adaptive Width** fails on this small dataset — it collapses to majority-class prediction (same as baseline), though it achieves the highest energy reduction (48.18%) trivially by exiting early on all samples.

---

### 2.2 ECG Arrhythmia (5-Class)

| Model | Params | MFLOPs | Accuracy (%) | Recall (%) | F1 (%) | ECE | Energy Red. (%) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **Base CNN (Control)** | 86,097 | 2.24 | **93.41** | **92.51** | **83.32** | **0.0125** | 0.00 | 0.43 |
| **Constant Width** | 86,097 | 2.24 | 88.77 | 84.42 | 79.79 | 0.1796 | 7.81 | 0.51 |
| **Increasing Width** | 122,353 | 1.34 | 88.94 | 87.52 | 79.54 | 0.1452 | **25.69** | **0.25** |
| **Decreasing Width** | 128,305 | 6.22 | 88.05 | 89.18 | 76.51 | 0.1638 | 2.95 | 0.67 |
| **Adaptive Width** | 98,577 | 2.24 | 88.68 | 81.72 | 76.66 | 0.0924 | 17.11 | 0.36 |

**Key Findings (ECG)**:
- The **Base CNN outperforms all early-exit variants** in accuracy (93.41%), recall (92.51%), F1 (83.32%), and calibration (ECE = 0.0125).
- Early-exit models suffer a **4–5% accuracy drop** and **4–7% F1 drop** compared to the baseline — the exit branches are under-trained with only 1+1 epochs.
- **Increasing Width** provides the best energy-efficiency tradeoff among early-exit models: 25.69% energy reduction with the lowest latency (0.25 ms), at only a ~4.5% accuracy cost.
- Calibration degrades substantially with early exits (ECE jumps from 0.0125 to 0.09–0.18).

---

### 2.3 MIT-BIH Arrhythmia (5-Class)

| Model | Params | MFLOPs | Accuracy (%) | Recall (%) | F1 (%) | ECE | Energy Red. (%) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **Base CNN (Control)** | 86,097 | 2.24 | **94.03** | **89.90** | **82.11** | **0.0125** | 0.00 | 0.43 |
| **Constant Width** | 86,097 | 2.24 | 91.07 | 69.94 | 66.60 | 0.2581 | 8.17 | 0.54 |
| **Increasing Width** | 122,353 | 1.34 | 89.78 | 77.95 | 72.07 | 0.2445 | **33.60** | **0.31** |
| **Decreasing Width** | 128,305 | 6.22 | 92.06 | 73.92 | 71.36 | 0.2529 | 2.27 | 0.67 |
| **Adaptive Width** | 98,577 | 2.24 | 89.11 | 85.99 | 73.00 | 0.1851 | 14.14 | 0.34 |

**Key Findings (MIT-BIH)**:
- Again, the **Base CNN outperforms all early-exit variants** — 94.03% accuracy, 82.11% F1, and excellent calibration (ECE = 0.0125).
- MIT-BIH shows the **worst early-exit degradation** of all three datasets: Constant Width F1 drops by **15.5 percentage points** (82.11% → 66.60%).
- **Increasing Width** again leads in efficiency: 33.60% energy reduction, lowest latency (0.31 ms), but at a 10-point F1 cost.
- Calibration is very poor across all early-exit models (ECE = 0.18–0.26), indicating premature exits with miscalibrated confidence.

---

## 3. Cross-Dataset Comparison

### 3.1 Accuracy Comparison

| Model | Bonn EEG | ECG | MIT-BIH |
|---|---|---|---|
| Base CNN (Control) | 75.00 | **93.41** | **94.03** |
| Constant Width | **93.33** | 88.77 | 91.07 |
| Increasing Width | **93.33** | 88.94 | 89.78 |
| Decreasing Width | **93.33** | 88.05 | 92.06 |
| Adaptive Width | 75.00 | 88.68 | 89.11 |

### 3.2 F1 Score Comparison

| Model | Bonn EEG | ECG | MIT-BIH |
|---|---|---|---|
| Base CNN (Control) | 42.86 | **83.32** | **82.11** |
| Constant Width | **90.68** | 79.79 | 66.60 |
| Increasing Width | **90.68** | 79.54 | 72.07 |
| Decreasing Width | **90.68** | 76.51 | 71.36 |
| Adaptive Width | 42.86 | 76.66 | 73.00 |

### 3.3 Energy Reduction Comparison (%)

| Model | Bonn EEG | ECG | MIT-BIH |
|---|---|---|---|
| Base CNN (Control) | 0.00 | 0.00 | 0.00 |
| Constant Width | 29.74 | 7.81 | 8.17 |
| Increasing Width | **42.11** | **25.69** | **33.60** |
| Decreasing Width | 15.83 | 2.95 | 2.27 |
| Adaptive Width | 48.18* | 17.11 | 14.14 |

*\* Adaptive Width on Bonn achieves high energy reduction trivially (model collapsed to majority-class prediction)*

### 3.4 Calibration (ECE) Comparison — Lower is Better

| Model | Bonn EEG | ECG | MIT-BIH |
|---|---|---|---|
| Base CNN (Control) | 0.2449 | **0.0125** | **0.0125** |
| Constant Width | 0.0641 | 0.1796 | 0.2581 |
| Increasing Width | 0.1086 | 0.1452 | 0.2445 |
| Decreasing Width | **0.0257** | 0.1638 | 0.2529 |
| Adaptive Width | 0.2434 | 0.0924 | 0.1851 |

---

## 4. Architecture Rankings

### Best Overall Accuracy
| Rank | Bonn EEG | ECG | MIT-BIH |
|---|---|---|---|
| 1st | Constant / Increasing / Decreasing (tied: 93.33%) | Base CNN (93.41%) | Base CNN (94.03%) |
| 2nd | — | Increasing Width (88.94%) | Decreasing Width (92.06%) |
| 3rd | Base CNN / Adaptive (tied: 75.00%) | Constant Width (88.77%) | Constant Width (91.07%) |

### Best Efficiency (Energy Reduction)
| Rank | Bonn EEG | ECG | MIT-BIH |
|---|---|---|---|
| 1st | Increasing Width (42.11%) | Increasing Width (25.69%) | Increasing Width (33.60%) |
| 2nd | Constant Width (29.74%) | Adaptive Width (17.11%) | Adaptive Width (14.14%) |
| 3rd | Decreasing Width (15.83%) | Constant Width (7.81%) | Constant Width (8.17%) |

### Best Calibration (Lowest ECE)
| Rank | Bonn EEG | ECG | MIT-BIH |
|---|---|---|---|
| 1st | Decreasing Width (0.0257) | Base CNN (0.0125) | Base CNN (0.0125) |
| 2nd | Constant Width (0.0641) | Adaptive Width (0.0924) | Adaptive Width (0.1851) |
| 3rd | Increasing Width (0.1086) | Increasing Width (0.1452) | Increasing Width (0.2445) |

---

## 5. Key Takeaways

### 5.1 Early Exits Help Most on Small, Simple Datasets
On the **Bonn EEG** dataset (280 samples, binary classification), early-exit architectures dramatically outperform the baseline — improving F1 from 42.86% to 90.68%. The joint training procedure with energy-aware loss appears to act as a strong regularizer that prevents overfitting on this tiny dataset. On the larger ECG/MIT-BIH datasets, the baseline CNN without early exits actually performs better, likely because early-exit branches are under-trained with only 1+1 epochs.

### 5.2 Increasing Width is Consistently the Most Efficient Architecture
Across **all three datasets**, the Increasing Width architecture `[32, 64, 128]` achieves the **highest energy reduction** (25–42%) and **lowest latency**. This makes intuitive sense: narrow early stages are cheap to compute, and easy samples exit before reaching the expensive later stages. This architecture is the clear winner for deployment on resource-constrained devices.

### 5.3 Training Budget Matters Critically
The stark contrast between Bonn (20 epochs) and ECG/MIT-BIH (2 epochs) results suggests that **early-exit branches require sufficient training**. With only 2 epochs:
- Exit branch classifiers are poorly calibrated (ECE >> 0.1)
- The model routes difficult samples to early exits incorrectly, degrading overall accuracy
- The energy-accuracy Pareto frontier is suboptimal

> **IMPORTANT:** The ECG and MIT-BIH experiments should be re-run with more training epochs (>=10 warmup + 10 joint) to enable fair comparison. The current 1+1 epoch results underestimate the potential of early-exit architectures on these datasets.

### 5.4 Adaptive Width Needs More Data or Training
The Adaptive Width model (with learned channel gating) **collapses on the Bonn dataset** due to its tiny size (280 samples). On the larger datasets, it performs reasonably but doesn't outperform simpler static architectures. The gating mechanism likely needs:
- More training epochs to learn meaningful sparsity patterns
- Larger datasets to avoid gate collapse

### 5.5 Calibration Is a Major Concern
Early-exit models consistently show **worse calibration** (higher ECE) compared to the baseline on ECG and MIT-BIH. This is problematic for clinical applications where reliable confidence estimates are critical. The **Decreasing Width** architecture shows the best calibration on Bonn (ECE = 0.0257), but this advantage disappears on the larger datasets.

### 5.6 Decreasing Width is Compute-Heavy with Minimal Efficiency Gains
The Decreasing Width architecture `[128, 64, 32]` has the **highest MFLOPs** (6.22 for ECG/MIT-BIH, 112.74 for Bonn) and achieves only **2–16% energy reduction** — the worst among early-exit variants. The expensive first stage means most compute is spent before any exit decision is made.

---

## 6. Recommendations

| Goal | Recommended Architecture | Rationale |
|---|---|---|
| **Maximum accuracy** | Base CNN (large datasets) or Constant/Increasing Width (small datasets) | Baselines win on well-trained large datasets; early exits help regularize small ones |
| **Maximum efficiency** | Increasing Width `[32, 64, 128]` | Consistently 25–42% energy reduction across all datasets |
| **Best calibration** | Decreasing Width (small data) or Base CNN (large data) | Trade-off: Decreasing Width has good ECE but poor efficiency |
| **Balanced (accuracy + efficiency)** | Increasing Width `[32, 64, 128]` | Best Pareto efficiency across all metrics |
| **Edge / IoT deployment** | Increasing Width with pruning | Combine architectural efficiency with structured pruning for further gains |

---

## 7. Experimental Configuration Summary

| Parameter | Bonn EEG | ECG | MIT-BIH |
|---|---|---|---|
| Batch Size | 64 | 128 | 128 |
| Learning Rate | 0.001 | 0.001 | 0.001 |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 |
| Energy Lambda | 0.02 | 0.02 | 0.02 |
| Warmup Epochs | 10 | 1 | 1 |
| Joint Epochs | 10 | 1 | 1 |
| Threshold Strategy | Confidence | Confidence | Confidence |
| Trials | 1 | 1 | 1 |
| Seed | 42 | 42 | 42 |

> **WARNING:** All experiments were run with only **1 trial** (no repeated measurements). The +/-0.00 standard deviations in the results tables reflect this — they are **not** indicators of zero variance. Multiple trials (>=5) with different seeds are needed for statistically significant conclusions.
