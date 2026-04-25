"""
quick_validate.py — Fast validation of the entire research pipeline.

Runs all experiments with minimal epochs/trials to verify correctness,
then produces results and plots. Use main.py --run-all for full research runs.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import torch.optim as optim

from dataset import get_dataloaders
from models import (GenericEarlyExitNet, AdaptiveEarlyExitNet,
                    EnergyJointLoss, AdaptiveEnergyJointLoss,
                    apply_structured_pruning)
from train import set_seed, train_classifiers_only, train_joint, calibrate_thresholds
from evaluate import inspect_data, evaluate_model_advanced
from analysis import (collect_exit_statistics, analyze_exit_patterns,
                      compute_difficulty_scores, print_analysis_report)
from visualize import generate_all_plots
from statistical_tests import compute_all_statistics, print_statistical_report

# ── Quick config: low epochs, 2 trials ──
WARMUP_EPOCHS = 3
JOINT_EPOCHS = 3
NUM_TRIALS = 2
SEED = 42
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4
ENERGY_LAMBDA = 0.02
SPARSITY_LAMBDA = 0.01
DATA_DIR = "./data/bonn"
PLOTS_DIR = "./plots"
RESULTS_DIR = "./results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data ──
train_dl, val_dl, test_dl, class_weights = get_dataloaders(DATA_DIR, BATCH_SIZE, use_freq_bands=True)
class_weights = class_weights.to(device)
inspect_data(train_dl)

# ── Architecture configs ──
arch_configs = {
    "Base CNN (Control)": {"channels": [64, 64, 64], "is_baseline": True, "adaptive": False},
    "Constant Width":     {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False},
    "Increasing Width":   {"channels": [32, 64, 128], "is_baseline": False, "adaptive": False},
    "Decreasing Width":   {"channels": [128, 64, 32], "is_baseline": False, "adaptive": False},
    "Adaptive Width":     {"channels": [64, 64, 64], "is_baseline": False, "adaptive": True},
}

def build_model(cfg):
    if cfg.get("adaptive"):
        return AdaptiveEarlyExitNet(in_channels=6, channel_sizes=cfg["channels"],
                                    num_classes=2, seq_len=4097).to(device)
    return GenericEarlyExitNet(in_channels=6, channel_sizes=cfg["channels"],
                               num_classes=2, seq_len=4097).to(device)

def get_criterion_fn(model, cfg):
    def criterion_fn(energy_lambda):
        if cfg.get("adaptive"):
            return AdaptiveEnergyJointLoss(model.stage_flops, class_weights,
                                           energy_lambda=energy_lambda,
                                           sparsity_lambda=SPARSITY_LAMBDA,
                                           is_baseline=cfg["is_baseline"])
        return EnergyJointLoss(model.stage_flops, class_weights,
                                energy_lambda=energy_lambda,
                                is_baseline=cfg["is_baseline"])
    return criterion_fn

def train_model(model, cfg, seed):
    set_seed(seed)
    criterion_fn = get_criterion_fn(model, cfg)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    total_epochs = WARMUP_EPOCHS + JOINT_EPOCHS

    if cfg["is_baseline"]:
        train_classifiers_only(model, train_dl, total_epochs, optimizer, criterion_fn, device,
                               scheduler=scheduler, max_grad_norm=1.0)
        return {"strategy": "confidence", "thresholds": [1.0] * (model.num_stages - 1)}
    else:
        train_classifiers_only(model, train_dl, WARMUP_EPOCHS, optimizer, criterion_fn, device,
                               scheduler=scheduler, max_grad_norm=1.0)
        train_joint(model, train_dl, JOINT_EPOCHS, optimizer, criterion_fn, device,
                    energy_lambda=ENERGY_LAMBDA, scheduler=scheduler, max_grad_norm=1.0)
        return calibrate_thresholds(model, val_dl, device, strategy="confidence", target_acc=0.95)


# =============================================
# 1. ARCHITECTURE ABLATION (fast)
# =============================================
print("\n" + "=" * 60)
print("EXPERIMENT 1: ARCHITECTURE ABLATION")
print("=" * 60)

results = []
all_trial_metrics = {}
viz_data = {"per_model": {}, "exit_distributions": {}}

for name, cfg in arch_configs.items():
    print(f"\n--- {name} ({NUM_TRIALS} trials) ---")
    trial_metrics = {k: [] for k in ['acc', 'recall', 'f1', 'ece', 'energy_red', 'latency']}

    for trial in range(NUM_TRIALS):
        trial_seed = SEED + trial
        print(f"  Trial {trial+1}/{NUM_TRIALS} (seed={trial_seed})")
        model = build_model(cfg)
        threshold_info = train_model(model, cfg, trial_seed)
        acc, recall, f1, ece, energy_red, latency, per_sample, ece_bins = \
            evaluate_model_advanced(model, test_dl, threshold_info, device, is_baseline=cfg["is_baseline"])

        trial_metrics['acc'].append(acc * 100)
        trial_metrics['recall'].append(recall * 100)
        trial_metrics['f1'].append(f1 * 100)
        trial_metrics['ece'].append(ece)
        trial_metrics['energy_red'].append(energy_red * 100)
        trial_metrics['latency'].append(latency)
        print(f"    Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%, ER: {energy_red*100:.2f}%")

    # Last-trial analysis
    exit_df = collect_exit_statistics(model, test_dl, threshold_info, device, is_baseline=cfg["is_baseline"])
    analysis = analyze_exit_patterns(exit_df, model.num_stages)
    print_analysis_report(analysis, name)

    all_trial_metrics[name] = trial_metrics
    total_params = model.count_parameters()
    total_flops = sum(model.stage_flops) / 1e6

    results.append({
        "Model": name,
        "Params": total_params,
        "MFLOPs": f"{total_flops:.2f}",
        "Accuracy (%)": f"{np.mean(trial_metrics['acc']):.2f} +/- {np.std(trial_metrics['acc']):.2f}",
        "Recall (%)": f"{np.mean(trial_metrics['recall']):.2f} +/- {np.std(trial_metrics['recall']):.2f}",
        "F1 Score (%)": f"{np.mean(trial_metrics['f1']):.2f} +/- {np.std(trial_metrics['f1']):.2f}",
        "ECE": f"{np.mean(trial_metrics['ece']):.4f} +/- {np.std(trial_metrics['ece']):.4f}",
        "Energy Red (%)": f"{np.mean(trial_metrics['energy_red']):.2f} +/- {np.std(trial_metrics['energy_red']):.2f}",
    })

    stage_exits = [int((exit_df["exit_stage"] == s).sum()) for s in range(model.num_stages)]
    viz_data["exit_distributions"][name] = stage_exits
    viz_data["per_model"][name] = {
        "ece_bin_data": ece_bins, "ece": np.mean(trial_metrics['ece']),
        "analysis": analysis, "exit_df": exit_df,
    }

df = pd.DataFrame(results)
viz_data["results_df"] = df
viz_data["results_list"] = results
print("\n" + df.to_string())

# Statistical tests
print("\n--- Statistical Significance Tests ---")
stats_results = compute_all_statistics(all_trial_metrics, list(all_trial_metrics.keys()))
print_statistical_report(stats_results)


# =============================================
# 2. THRESHOLD STRATEGY COMPARISON
# =============================================
print("\n" + "=" * 60)
print("EXPERIMENT 2: THRESHOLD STRATEGY COMPARISON")
print("=" * 60)

strategies = ["confidence", "entropy", "patience"]
strategy_results = {}

cfg = {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False}
set_seed(SEED)
model = build_model(cfg)
train_model(model, cfg, SEED)

for strategy in strategies:
    print(f"\n--- Strategy: {strategy} ---")
    threshold_info = calibrate_thresholds(model, val_dl, device, strategy=strategy,
                                           target_acc=0.95, entropy_percentile=80, patience=2)
    acc, recall, f1, ece, energy_red, latency, _, _ = \
        evaluate_model_advanced(model, test_dl, threshold_info, device, is_baseline=False)
    strategy_results[strategy] = {
        "accuracy": acc * 100, "f1": f1 * 100,
        "energy_reduction": energy_red * 100, "ece": ece, "latency_ms": latency,
    }
    print(f"    Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%, ER: {energy_red*100:.2f}%")

viz_data["strategy_results"] = strategy_results


# =============================================
# 3. MODEL SIZE SCALING
# =============================================
print("\n" + "=" * 60)
print("EXPERIMENT 3: MODEL SIZE SCALING")
print("=" * 60)

size_configs = {
    "Tiny [16]":   [16, 16, 16],
    "Small [32]":  [32, 32, 32],
    "Medium [64]": [64, 64, 64],
    "Large [128]": [128, 128, 128],
}
size_results = []

for name, channels in size_configs.items():
    cfg = {"channels": channels, "is_baseline": False, "adaptive": False}
    model = build_model(cfg)
    params = model.count_parameters()
    flops = sum(model.stage_flops) / 1e6
    print(f"\n--- {name}: {params:,} params, {flops:.2f} MFLOPs ---")
    set_seed(SEED)
    threshold_info = train_model(model, cfg, SEED)
    acc, recall, f1, ece, energy_red, latency, _, _ = \
        evaluate_model_advanced(model, test_dl, threshold_info, device, is_baseline=False)
    size_results.append({
        "name": name, "channels": channels, "params": params, "mflops": flops,
        "accuracy": acc * 100, "f1": f1 * 100, "energy_reduction": energy_red * 100,
    })
    print(f"    Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%, ER: {energy_red*100:.2f}%")

viz_data["size_results"] = size_results
print("\n" + pd.DataFrame(size_results).to_string())


# =============================================
# 4. STRUCTURED PRUNING
# =============================================
print("\n" + "=" * 60)
print("EXPERIMENT 4: STRUCTURED PRUNING")
print("=" * 60)

prune_ratios = [0.0, 0.25, 0.5]
pruning_results = []
cfg = {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False}
set_seed(SEED)
base_model = build_model(cfg)
train_model(base_model, cfg, SEED)
base_flops = sum(base_model.stage_flops)

for ratio in prune_ratios:
    print(f"\n--- Prune Ratio: {ratio:.0%} ---")
    if ratio == 0.0:
        model = base_model
        pruned_flops = base_flops
    else:
        model, _ = apply_structured_pruning(base_model, prune_ratio=ratio,
                                             in_channels=6, seq_len=4097, num_classes=2)
        model = model.to(device)
        pruned_flops = sum(model.stage_flops)
        criterion_fn = get_criterion_fn(model, cfg)
        optimizer = optim.Adam(model.parameters(), lr=LR * 0.1)
        train_joint(model, train_dl, 2, optimizer, criterion_fn, device, energy_lambda=ENERGY_LAMBDA)

    threshold_info = calibrate_thresholds(model, val_dl, device, strategy="confidence")
    acc, recall, f1, ece, energy_red, latency, _, _ = \
        evaluate_model_advanced(model, test_dl, threshold_info, device, is_baseline=False)
    flops_red = (1.0 - pruned_flops / base_flops) * 100
    pruning_results.append({
        "prune_ratio": ratio, "params": model.count_parameters(),
        "accuracy": acc * 100, "f1": f1 * 100,
        "flops_reduction": flops_red, "energy_reduction": energy_red * 100,
    })
    print(f"    Acc: {acc*100:.2f}%, FLOPs Red: {flops_red:.1f}%")

viz_data["pruning_results"] = pruning_results


# =============================================
# 5. HYPERPARAMETER SENSITIVITY (tiny grid)
# =============================================
print("\n" + "=" * 60)
print("EXPERIMENT 5: HYPERPARAMETER SENSITIVITY")
print("=" * 60)

hp_results = []
for lr in [5e-4, 1e-3]:
    for el in [0.01, 0.05]:
        print(f"\n  lr={lr}, energy_lambda={el}")
        set_seed(SEED)
        cfg = {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False}
        model = build_model(cfg)
        criterion_fn = get_criterion_fn(model, cfg)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        train_classifiers_only(model, train_dl, WARMUP_EPOCHS, optimizer, criterion_fn, device)
        train_joint(model, train_dl, JOINT_EPOCHS, optimizer, criterion_fn, device, energy_lambda=el)
        threshold_info = calibrate_thresholds(model, val_dl, device, strategy="confidence")
        acc, recall, f1, ece, energy_red, latency, _, _ = \
            evaluate_model_advanced(model, test_dl, threshold_info, device, is_baseline=False)
        hp_results.append({
            "lr": lr, "energy_lambda": el,
            "accuracy": acc * 100, "f1": f1 * 100, "energy_reduction": energy_red * 100,
        })
        print(f"    Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%, ER: {energy_red*100:.2f}%")

viz_data["tuning_results"] = hp_results


# =============================================
# GENERATE ALL PLOTS + SAVE RESULTS
# =============================================
generate_all_plots(viz_data, PLOTS_DIR)

# Save CSV
df.to_csv(os.path.join(RESULTS_DIR, "main_results.csv"), index=False)

# Save comprehensive JSON
json_output = {
    "config": {
        "warmup_epochs": WARMUP_EPOCHS, "joint_epochs": JOINT_EPOCHS,
        "num_trials": NUM_TRIALS, "seed": SEED, "lr": LR,
        "energy_lambda": ENERGY_LAMBDA, "batch_size": BATCH_SIZE,
    },
    "architecture_ablation": results,
    "raw_trial_metrics": {
        name: {k: [float(v) for v in vals] for k, vals in metrics.items()}
        for name, metrics in all_trial_metrics.items()
    },
    "threshold_strategies": strategy_results,
    "model_sizes": size_results,
    "pruning": pruning_results,
    "hp_sensitivity": hp_results,
}
with open(os.path.join(RESULTS_DIR, "experiment_results.json"), 'w') as f:
    json.dump(json_output, f, indent=2, default=str)

print("\n" + "=" * 60)
print("[DONE] All experiments complete.")
print(f"  Results: {RESULTS_DIR}/")
print(f"  Plots:   {PLOTS_DIR}/")
print("=" * 60)
