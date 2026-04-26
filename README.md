# Early-Exit Neural Network Architecture Research
### Energy-Efficient Seizure Detection from EEG Signals

**Rithwak Somepalli · Suryaprakash Murugavvel · Monique Gaye · Amrutha Kodali**

---

## Overview

Traditional deep neural networks process every input through the entire network, regardless of difficulty. For seizure detection on wearable EEG devices, this approach expends equal energy on both obvious non-seizure cases and ambiguous edge cases. This inefficiency is particularly significant in real-world healthcare applications, where resource optimization is crucial.

This project investigates **early-exit neural network architectures** for seizure detection: models that can stop and produce a prediction as soon as confidence is high enough, skipping unnecessary computation. In addition to building on existing methods, we introduce two novel architectures, decreasing-width and adaptive-width, and conduct a comprehensive comparison of all five models in terms of accuracy, recall, F1 score, and energy savings (FLOPs).

---

## The Five Architectures

| # | Architecture | Description |
|---|---|---|
| 1 | **Baseline CNN** | Standard deep CNN, no early exit. Full-depth reference point. |
| 2 | **Constant-Width Early Exit** | Uniform layer width with exit heads at multiple depths. |
| 3 | **Increasing-Width Early Exit** | Progressively wider layers with exit heads. |
| 4 | **Decreasing-Width Early Exit** *(novel)* | Funnel design — wider early layers capture broad EEG features, narrower deeper layers refine. |
| 5 | **Adaptive-Width Early Exit** *(novel)* | The network dynamically prunes irrelevant features as depth increases. |

The hypothesis behind the novel architectures: early layers need to capture broad patterns across EEG channels and frequency bands, while deeper layers should focus only on the most discriminative features. A narrowing structure may reduce memory and energy consumption without sacrificing accuracy.

---

## Current Results

> All results on the Bonn EEG seizure dataset (5 trials, seeds 42–46, confidence thresholding). Energy reduction measured via FLOPs-weighted early-exit savings.

| Model | Accuracy | Recall | F1 Score | Energy Reduction |
|---|---|---|---|---|
| Baseline CNN (Control) | 95.33 ± 2.87% | 92.89 ± 6.83% | 93.33 ± 4.72% | 0% |
| Constant-Width EE | 83.67 ± 7.41% | 67.78 ± 15.41% | 67.43 ± 18.60% | 24.04 ± 2.70% |
| **Increasing-Width EE** ⭐ | **94.00 ± 2.26%** | **91.56 ± 4.13%** | **91.86 ± 3.13%** | **42.05 ± 2.99%** |
| Decreasing-Width EE | 88.33 ± 3.33% | 83.78 ± 5.33% | 84.13 ± 4.20% | 12.84 ± 1.92% |
| Adaptive-Width EE | 86.00 ± 6.38% | 73.78 ± 13.31% | 75.32 ± 14.11% | 47.48 ± 2.55% |

---

## Repo Structure

```
Early_exit_architecture_research/
│
├── data/bonn/              # Bonn seizure dataset (EEG samples)
├── plots/                  # Generated visualizations and metric plots
│
├── main.py                 # Entry point — runs full training + evaluation pipeline
├── models.py               # All 5 architecture definitions (CNN, constant, increasing, decreasing, adaptive)
├── dataset.py              # Data loading, preprocessing, normalization, class balancing
├── train.py                # Training loop with exit-head logic
├── evaluate.py             # Accuracy, F1, recall, FLOPs, exit distribution metrics
├── analysis.py             # Exit behavior analysis and cross-dataset evaluation
├── visualize.py            # Plot generation: trade-off curves, calibration plots, exit histograms
├── tuning.py               # Hyperparameter search utilities
│
├── status_report.ipynb     # Jupyter notebook with current partial results
├── status.md               # Running status notes
└── .env.example            # Environment variable template
```

---
 
## Getting Started
 
> **Recommended:** Run on Google Colab for free GPU access and zero setup. Local instructions are below if you prefer.
 
---
 
### Option A — Google Colab *(recommended)*
 
**Step 1.** Download the Bonn seizure dataset from [Google Drive](https://drive.google.com/drive/folders/132mm7W9rXuarB2wd_cMy4n9-qeTmIJFX?usp=sharing) and add it to your own Drive.
 
**Step 2.** Open a notebook and update the path in the first cell to match your Drive location.
 
**Step 3.** Click **Run All** — all metrics and plots will appear at the bottom.
 
| Notebook | Purpose |
|---|---|
| [Preprocessing](https://colab.research.google.com/drive/1cZWTCWrdIi9rCaKxJfOZ2dzz9EKVXI86?usp=sharing) | Data loading, normalization, class balancing |
| [Implementation & Results](https://colab.research.google.com/drive/1lwuYQafT6nnBWmTOMrky1WJo3uo7_UbN?usp=sharing) | Model training, evaluation, and plots |
 
---
 
### Option B — Run Locally
 
**Prerequisites:** Python 3.8+, pip, a CUDA-capable GPU (recommended)
 
```bash
# 1. Clone the repo
git clone https://github.com/Srithwak/Early_exit_architecture_research.git
cd Early_exit_architecture_research
 
# 2. Install dependencies
pip install torch numpy pandas matplotlib scikit-learn
 
# 3. Add the dataset
#    Download from the Drive link above and place it at:
#    data/bonn/
 
# 4. Run
python main.py
```
 
Output — trained models, evaluation metrics, and all plots — will be saved to `plots/`.

---

## Evaluation Metrics

Each architecture is benchmarked on:

- **Accuracy** — Overall classification correctness
- **Recall** — Critical for a medical use case; missed seizures are costly
- **F1 Score** — Harmonic mean of precision and recall
- **Exit Distribution** — At which exit heads do samples leave? What inputs trigger early exits?
- **Energy Savings (FLOPs)** — Floating-point operations per inference vs. baseline
- **Confidence Calibration** — Are the model's confidence scores reliable?

---

## Datasets

| Dataset | Channels | Sample Rate | Notes |
|---|---|---|---|
| [Bonn Seizure Dataset](https://drive.google.com/drive/folders/132mm7W9rXuarB2wd_cMy4n9-qeTmIJFX?usp=sharing) | Single-channel | 173.6 Hz | Small, clean — fast iteration |
| [CHB-MIT (PhysioNet)](https://physionet.org/content/chbmit/1.0.0/) | 23–26 channels | 256 Hz | 983 hours, 198 seizures across 23 patients |

The Bonn dataset is the primary development dataset due to its manageable size. CHB-MIT will be used for cross-dataset evaluation and scaling experiments.

---

## Roadmap

- [x] Baseline CNN implemented and evaluated
- [x] Constant-width and increasing-width early-exit architectures
- [x] Decreasing-width early-exit architecture (novel)
- [x] Adaptive-width early-exit architecture (novel)
- [ ] Cross-dataset evaluation (train on Bonn → test on CHB-MIT)
- [x] Deeper exit behavior analysis (overthinking/underthinking, exit heatmaps, per-stage accuracy)
- [x] Hyperparameter tuning (grid search over LR × energy lambda × weight decay)
- [x] Advanced feature pruning — structured pruning (L1-norm channel removal) + channel gating
- [x] Trade-off curves (Pareto plots), calibration diagrams, exit distributions, confidence histograms
- [ ] Comparison against published benchmarks
- [x] Dynamic thresholding — confidence, entropy, and patience strategies implemented and compared
- [x] Statistical significance testing (paired t-tests, Wilcoxon, Cohen's d, 95% CIs)
- [x] Model size scaling experiment (Tiny → XLarge)

---

## References

[1] Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C. K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23). https://physionet.org/content/chbmit/1.0.0/
 
[2] American Epilepsy Society. (2014). American Epilepsy Society Seizure Prediction Challenge Dataset. Kaggle. https://www.kaggle.com/c/seizure-prediction/data
 
[3] PhysioNet. (2019). Siena Scalp EEG Dataset. https://physionet.org/content/siena-scalp-eeg/1.0.0/

---

## Team

| Name | Role |
|---|---|
| **Rithwak Somepalli** | Novel architecture implementation (decreasing-width, adaptive-width), feature pruning research |
| **Suryaprakash Murugavvel** | Baseline CNN + existing architectures, hyperparameter tuning, model evaluation |
| **Monique Gaye** | Research, energy/memory analysis, exit threshold analysis, visualizations |
| **Amrutha Kodali** | Data preprocessing, normalization, class balancing, cross-dataset evaluation |
