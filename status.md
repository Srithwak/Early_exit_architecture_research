# Early Exit Architecture Research — Status Report

> **Last Updated:** 2026-04-25 09:28 ET

---

## Project Status: Presentation-Ready

All experiments run end-to-end. Pipeline produces reproducible results, statistical tests, and publication-quality plots. Tuned Adaptive Width model shows +4.3pp accuracy and +11.6pp F1 improvement over defaults. Presentation guide, diagrams, and scripts are complete.

---

## Completed Components

### 1. Bias Audit & Reproducibility
- `set_seed(seed)` controls `torch`, `numpy`, `random`, CUDA, and cuDNN determinism
- Gradient clipping (`max_grad_norm=1.0`) in both training phases
- LR scheduler support (`ReduceLROnPlateau`)
- All models train under identical conditions (same total epochs, same seeds)
- Baseline gets same total epoch budget (all in warmup mode)
- Multi-trial averaging with mean +/- std reporting (default 5 trials)

### 2. Dynamic Thresholding (3 strategies)
- **Confidence**: exit when max softmax probability > threshold
- **Entropy**: exit when prediction entropy < threshold (calibrated from correct-prediction percentile)
- **Patience**: exit when N consecutive stages predict the same class
- `calibrate_thresholds()` returns strategy dict with metadata
- `_should_exit()` dispatcher handles all strategies
- `--strategy` CLI flag, `run_threshold_strategy_comparison()`

### 3. Deeper Exit Behavior Analysis
- `collect_exit_statistics()` — per-sample data including ALL stage predictions
- `analyze_exit_patterns()` — per-class exits, per-stage accuracy/confidence/entropy
- **Overthinking detection**: samples correct at earlier stage but exited later
- **Underthinking detection**: samples that exited early but incorrectly
- `compute_difficulty_scores()` — assigns difficulty based on earliest correct stage

### 4. Feature Pruning Mechanisms
- `compute_channel_importance()` — L1-norm importance scores per Conv1d channel
- `apply_structured_pruning()` — creates genuinely smaller network (not masked)
- `_transfer_pruned_weights()` — handles Conv1d + BatchNorm weight transfer
- Ablation over prune ratios [0%, 10%, 25%, 50%]
- Post-pruning fine-tuning with reduced learning rate

### 5. Model Size Experiments
- 5 model sizes: Tiny [16], Small [32], Medium [64], Large [128], XLarge [256]
- Each tested with identical training protocol
- Reports params, MFLOPs, accuracy, F1, energy reduction

### 6. Hyperparameter Tuning
- `HyperparameterSearchSpace` dataclass defining the grid
- `run_hyperparameter_search()` — grid search with random subsampling
- Supports both adaptive (with sparsity_lambda) and standard models
- Results saved as CSV, ranked by F1 score

### 7. Visualizations (10 plot types)
1. Accuracy vs Energy Pareto (with error bars)
2. Reliability / Calibration Diagram (per-model, with ECE)
3. Exit Distribution Heatmap (classes x stages)
4. Per-Stage Accuracy Breakdown
5. Confidence Distributions (correct vs incorrect)
6. Overthinking/Underthinking Chart
7. HP Sensitivity Heatmap (LR vs energy_lambda -> F1)
8. Model Size Scaling (dual-axis)
9. Pruning Impact (accuracy + FLOPs vs prune ratio)
10. Threshold Strategy Comparison (side-by-side bars)
- All plots saved as PNG (300 DPI) + PDF

### 8. Statistical Significance Testing (NEW)
- `statistical_tests.py` module with:
  - Paired t-tests between all model pairs
  - Wilcoxon signed-rank tests (non-parametric alternative)
  - Cohen's d effect sizes with magnitude interpretation
  - 95% confidence intervals using t-distribution
  - Comprehensive formatted reporting
- Automatically runs after architecture ablation
- Results saved in `experiment_results.json`

---

## File Structure

```
ML_project/
  main.py              — ResearchPipeline class + CLI (--run-all, etc.)
  quick_validate.py    — Fast validation script (3+3 epochs, 2 trials)
  models.py            — GenericEarlyExitNet, AdaptiveEarlyExitNet, losses, pruning
  train.py             — 2-phase training + 3 threshold calibration strategies
  evaluate.py          — Strategy-aware eval with ECE, latency, per-sample data
  analysis.py          — Overthinking/underthinking detection, difficulty scoring
  visualize.py         — 10 publication-quality plot types
  tuning.py            — Hyperparameter grid/random search
  statistical_tests.py — Paired t-tests, Wilcoxon, CIs, effect sizes
  dataset.py           — Bonn EEG loader with FFT frequency band features
  data/bonn/           — Preprocessed Bonn EEG data (280 train, 60 val, 60 test)
  results/             — CSV + JSON output
  plots/               — 64 plot files (PNG + PDF)
```

---

## CLI Usage

```bash
# Quick validation (~13 min on CPU)
python quick_validate.py

# Full research run (~40 min on CPU)
python main.py --run-all --trials 5 --seed 42

# Individual experiments
python main.py --run-ablation --trials 5
python main.py --run-strategies
python main.py --run-sizes
python main.py --run-pruning
python main.py --run-tuning

# Options
python main.py --run-ablation --seed 42 --trials 5 --strategy entropy
```

---

## Key Design Decisions (Unbiased Research)

1. All models trained under identical conditions (same total epochs, same seeds)
2. Baseline gets same total epoch budget, all in warmup mode
3. Multiple trials per configuration with mean +/- std reporting
4. Statistical significance tests (paired t-tests + Wilcoxon) between all model pairs
5. Class imbalance handled via weighted cross-entropy loss
6. Structured pruning creates genuinely smaller networks, not masked ones
7. Three orthogonal threshold strategies compared on the same trained model
