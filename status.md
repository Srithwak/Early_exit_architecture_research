# Early Exit Architecture Research — Status Report

> **Last Updated:** 2026-04-23 23:12 ET

---

## ✅ Completed

### 1. Bias Audit & Reproducibility ✅
**What was done:**
- Added `set_seed(seed)` to `train.py` — controls `torch`, `numpy`, `random`, CUDA, and cuDNN determinism
- Added gradient clipping (`max_grad_norm=1.0`) to both training phases
- Added LR scheduler support (`ReduceLROnPlateau`) to both training phases
- **Consolidated all 5 model configurations** into a single `ResearchPipeline` class in `main.py`
- **Deprecated `run_pipeline()`** — all experiments now run through `ResearchPipeline`
- **Fixed unequal training**: baseline models now train for `warmup_epochs + joint_epochs` total in warmup mode, matching the total epoch count of early-exit models
- **Fixed inconsistent `target_acc`**: unified to a configurable `ExperimentConfig.target_acc` (default 0.95)
- Multi-trial averaging with mean ± std reporting (default 3 trials)
- Per-trial seed control: `base_seed + trial_index`

### 2. Dynamic Thresholding ✅
**What was done:**
- Implemented 3 exit strategies in `train.py`:
  - **Confidence** (original): exit when max softmax probability > threshold
  - **Entropy**: exit when prediction entropy < threshold (calibrated from correct-prediction percentile)
  - **Patience**: exit when N consecutive stages predict the same class
- `calibrate_thresholds()` now returns a strategy dict with metadata instead of a bare list
- `evaluate.py` updated with `_should_exit()` dispatcher that handles all strategies
- Both `evaluate_model()` and `evaluate_model_advanced()` accept either old-style lists or new strategy dicts
- Added `--strategy` CLI flag to select threshold strategy
- Added `run_threshold_strategy_comparison()` to compare all strategies on same model

### 3. Deeper Exit Behavior Analysis ✅
**What was done:**
- Created `analysis.py` with:
  - `collect_exit_statistics()` — per-sample data including ALL stage predictions (not just exit stage)
  - `analyze_exit_patterns()` — per-class exits, per-stage accuracy/confidence/entropy, overthinking, underthinking
  - `compute_difficulty_scores()` — assigns difficulty based on earliest correct stage
  - `print_analysis_report()` — formatted summary
- **Overthinking detection**: samples correct at earlier stage but exited later
- **Underthinking detection**: samples that exited early but incorrectly
- Integrated into `ResearchPipeline.run_architecture_ablation()` — analysis runs on last trial of each model

### 4. Feature Pruning Mechanisms ✅
**What was done:**
- Added to `models.py`:
  - `compute_channel_importance()` — L1-norm importance scores per Conv1d channel
  - `apply_structured_pruning()` — creates genuinely smaller network (not masked), transfers surviving weights
  - `_transfer_pruned_weights()` — handles Conv1d + BatchNorm weight transfer
- Reports parameter reduction and FLOPs reduction after pruning
- Added `run_pruning_experiment()` to pipeline — ablates over prune ratios [0%, 10%, 25%, 50%]
- Post-pruning fine-tuning with reduced learning rate

### 5. Model Size Experiments ✅
**What was done:**
- Added 5 model sizes: Tiny [16,16,16], Small [32,32,32], Medium [64,64,64], Large [128,128,128], XLarge [256,256,256]
- Each tested with identical training protocol
- `count_parameters()` method added to `GenericEarlyExitNet`
- Added `run_model_size_experiment()` to pipeline
- Reports params, MFLOPs, accuracy, F1, energy reduction

### 6. Better Hyperparameter Tuning ✅
**What was done:**
- Created `tuning.py` with:
  - `HyperparameterSearchSpace` dataclass defining the grid
  - `run_hyperparameter_search()` — grid search with random subsampling for large grids
  - Supports both adaptive (with sparsity_lambda) and standard models
  - Results saved as CSV, ranked by F1 score
  - Top-5 configurations printed automatically
- Added `--run-tuning` CLI flag
- Default search space: lr × energy_lambda × weight_decay × warmup_epochs × joint_epochs

### 7. Stronger Visualizations ✅
**What was done:**
- Created `visualize.py` with 10 publication-quality plot types:
  1. **Accuracy vs Energy Pareto** — with error bars from multi-trial runs
  2. **Reliability / Calibration Diagram** — per-model, with ECE annotated
  3. **Exit Distribution Heatmap** — classes × stages, normalized percentages
  4. **Per-Stage Accuracy Breakdown** — bar chart with sample counts
  5. **Confidence Distributions** — correct vs incorrect, per exit stage
  6. **Overthinking/Underthinking Chart** — exit quality summary
  7. **HP Sensitivity Heatmap** — lr vs energy_lambda → F1
  8. **Model Size Scaling** — dual-axis accuracy + energy vs parameters
  9. **Pruning Impact** — accuracy and FLOPs reduction vs prune ratio
  10. **Threshold Strategy Comparison** — side-by-side bar charts
- All plots saved as PNG (300 DPI) + PDF
- Consistent seaborn-whitegrid style
- `generate_all_plots()` master function

---

## 📋 CLI Usage

```bash
# Run architecture ablation (default)
python main.py --run-ablation

# Run all experiments
python main.py --run-all

# Run specific experiments
python main.py --run-sizes          # Model size scaling
python main.py --run-pruning        # Structured pruning
python main.py --run-strategies     # Threshold strategy comparison
python main.py --run-tuning         # Hyperparameter search

# Options
python main.py --run-ablation --seed 42 --trials 5 --strategy entropy
```

---

## 🧪 Context for Future Conversations

This project is a research study on early exit neural network architectures for EEG seizure classification using the Bonn dataset. The core idea: attach intermediate classifiers to a CNN so "easy" samples can exit early, saving computation, while "hard" samples use the full network.

**Key files:**
- `dataset.py` — Loads Bonn EEG data with FFT frequency band features (6 channels: 1 raw + 5 frequency bands)
- `models.py` — `GenericEarlyExitNet`, `AdaptiveEarlyExitNet` (with ChannelGate), loss functions, **structured pruning**
- `train.py` — Two-phase training + **3 threshold calibration strategies** (confidence/entropy/patience)
- `evaluate.py` — Strategy-aware evaluation with ECE, latency, **per-sample data collection**
- `analysis.py` — **Per-sample exit behavior analysis**, overthinking/underthinking detection, difficulty scoring
- `tuning.py` — **Hyperparameter grid search** with CSV output
- `visualize.py` — **10 publication-quality plot types**
- `main.py` — `ResearchPipeline` class orchestrating all experiments with CLI interface

**Important design decisions:**
- All models trained under identical conditions (same total epochs, same seeds) for unbiased comparison
- Baseline gets same total epoch budget, all in warmup mode
- Multiple trials per configuration with mean ± std reporting
- Dynamic thresholding replaces fixed accuracy targets
- Structured pruning creates genuinely smaller networks, not masked ones
