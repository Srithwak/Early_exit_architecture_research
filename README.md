# Early-Exit Neural Network Architecture Research
### Energy-Efficient Biomedical Signal Classification

**Rithwak Somepalli · Suryaprakash Murugavvel · Monique Gaye · Amrutha Kodali**

---

## Overview

Traditional deep neural networks process every input through the entire network, regardless of difficulty. For real-time seizure and arrhythmia detection on wearable devices, this wastes energy on obvious cases that could be classified early.

This project investigates **early-exit neural network architectures** — models that can stop and produce a prediction as soon as confidence is high enough, skipping unnecessary computation. We compare five architectures (including two novel designs) across **three clinical datasets**, evaluating accuracy, F1 score, energy savings, calibration, and latency.

---

## The Five Architectures

| # | Architecture | Channels | Type |
|---|---|---|---|
| 1 | **Baseline CNN** | [64, 64, 64] | Control (no exits) |
| 2 | **Constant-Width EE** | [64, 64, 64] | Existing |
| 3 | **Increasing-Width EE** ⭐ | [32, 64, 128] | Existing |
| 4 | **Decreasing-Width EE** *(novel)* | [128, 64, 32] | Novel |
| 5 | **Adaptive-Width EE** *(novel)* | [64, 64, 64] + gates | Novel |

---

## Results Summary

### Bonn EEG (Seizure Detection — Binary, 280 train samples)

| Model | Accuracy | F1 Score | Energy Red. | ECE |
|---|---|---|---|---|
| Baseline CNN | 75.00% | 42.86% | 0% | 0.2449 |
| Constant Width | **93.33%** | **90.68%** | 29.74% | 0.0641 |
| **Increasing Width** ⭐ | **93.33%** | **90.68%** | **42.11%** | 0.1086 |
| Decreasing Width | **93.33%** | **90.68%** | 15.83% | **0.0257** |
| Adaptive Width | 75.00% | 42.86% | 48.18%* | 0.2434 |

*\*Adaptive Width collapsed to majority-class prediction on this small dataset*

### ECG Arrhythmia (5-class, 87K train samples)

| Model | Accuracy | F1 Score | Energy Red. | ECE |
|---|---|---|---|---|
| **Baseline CNN** | **93.41%** | **83.32%** | 0% | **0.0125** |
| Constant Width | 88.77% | 79.79% | 7.81% | 0.1796 |
| **Increasing Width** ⭐ | 88.94% | 79.54% | **25.69%** | 0.1452 |
| Decreasing Width | 88.05% | 76.51% | 2.95% | 0.1638 |
| Adaptive Width | 88.68% | 76.66% | 17.11% | 0.0924 |

### MIT-BIH Arrhythmia (5-class, 77K train samples)

| Model | Accuracy | F1 Score | Energy Red. | ECE |
|---|---|---|---|---|
| **Baseline CNN** | **94.03%** | **82.11%** | 0% | **0.0125** |
| Constant Width | 91.07% | 66.60% | 8.17% | 0.2581 |
| **Increasing Width** ⭐ | 89.78% | 72.07% | **33.60%** | 0.2445 |
| Decreasing Width | 92.06% | 71.36% | 2.27% | 0.2529 |
| Adaptive Width | 89.11% | 73.00% | 14.14% | 0.1851 |

> **Key Finding:** Increasing Width `[32, 64, 128]` achieves the highest energy reduction (25–42%) across *all three datasets* with the lowest latency. On small datasets, early exits also act as regularizers, dramatically improving F1.

See [`cross_dataset_analysis.md`](cross_dataset_analysis.md) for the full analysis.

---

## Repo Structure

```
Early_exit_architecture_research/
├── main.py                 # Bonn EEG pipeline (CLI entry point)
├── main_ecg.py             # ECG Arrhythmia pipeline
├── main_mitbih.py          # MIT-BIH Arrhythmia pipeline
├── models.py               # All 5 architectures + losses + pruning
├── models_mitbih.py        # MIT-BIH model variants
├── dataset.py              # Bonn EEG data loader
├── dataset_ecg.py          # ECG data loader
├── dataset_mitbih.py       # MIT-BIH data loader
├── train.py                # 2-phase training + threshold calibration
├── evaluate.py             # Evaluation with ECE, latency, per-sample tracking
├── analysis.py             # Exit behavior analysis (overthinking/underthinking)
├── visualize.py            # Plot generation (10 plot types, PNG + PDF)
├── statistical_tests.py    # Paired t-tests, Wilcoxon, Cohen's d, CIs
├── tuning.py               # Hyperparameter search utilities
├── tune_adaptive.py        # Adaptive Width tuning script
├── cross_dataset_analysis.md  # Full cross-dataset results & analysis
├── presentation.md         # Presentation slides & script
├── requirements.txt        # Python dependencies
├── setup_venv.sh           # Virtual environment setup script
└── create_nb_*.py          # Notebook generation scripts
```

---

## Getting Started

### Option A — Google Colab *(recommended)*

**Step 1.** Download the dataset from [Google Drive](https://drive.google.com/drive/folders/132mm7W9rXuarB2wd_cMy4n9-qeTmIJFX?usp=sharing) and add it to your Drive.

**Step 2.** Open the notebook and update the data path.

**Step 3.** Click **Run All**.

| Notebook | Purpose |
|---|---|
| [Preprocessing](https://colab.research.google.com/drive/1cZWTCWrdIi9rCaKxJfOZ2dzz9EKVXI86?usp=sharing) | Data loading, normalization, class balancing |
| [Implementation & Results](https://colab.research.google.com/drive/1lwuYQafT6nnBWmTOMrky1WJo3uo7_UbN?usp=sharing) | Model training, evaluation, and plots |

---

### Option B — Run Locally

**Prerequisites:** Python 3.10+, pip

```bash
# 1. Clone the repo
git clone https://github.com/Srithwak/Early_exit_architecture_research.git
cd Early_exit_architecture_research

# 2. Set up virtual environment & install dependencies
chmod +x setup_venv.sh
./setup_venv.sh
source venv/bin/activate

# 3. Add datasets to data/bonn/, data/ecg/, data/mitbih/

# 4. Run experiments
python main.py --run-all --trials 5              # Bonn EEG
python main_ecg.py --run-all --trials 1           # ECG Arrhythmia
python main_mitbih.py --run-all --trials 1        # MIT-BIH
```

Output (metrics, plots, JSON results) is saved to `results*/` and `plots*/` directories.

---

## Datasets

| Dataset | Samples | Seq Len | Classes | Domain |
|---|---|---|---|---|
| **Bonn EEG** | 400 | 4,097 | 2 | EEG Seizure Detection |
| **ECG Arrhythmia** | ~109K | 187 | 5 | ECG Classification |
| **MIT-BIH Arrhythmia** | ~96K | 187 | 5 | ECG Classification |

All datasets use 6-channel input (1 raw signal + 5 FFT frequency bands).

---

## CLI Options

```bash
python main.py [OPTIONS]

--run-ablation    Architecture ablation study
--run-sizes       Model size scaling experiment
--run-pruning     Structured pruning experiment
--run-strategies  Compare threshold strategies
--run-tuning      Hyperparameter search
--run-all         Run all experiments
--trials N        Number of trials (default: 3)
--seed S          Base random seed (default: 42)
--strategy STR    Threshold strategy: confidence|entropy|patience
```

---

## References

[1] Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23). https://physionet.org/

[2] Bonn EEG Dataset. University of Bonn, Department of Epileptology.

[3] Moody, G. B. & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3).

---

## Team

| Name | Role |
|---|---|
| **Rithwak Somepalli** | Novel architecture design & implementation, structured pruning, pipeline development, statistical testing |
| **Suryaprakash Murugavvel** | Baseline + existing architectures, hyperparameter tuning, model evaluation |
| **Monique Gaye** | Energy/memory analysis, exit threshold strategies, visualizations |
| **Amrutha Kodali** | Data preprocessing, normalization, cross-dataset evaluation |
