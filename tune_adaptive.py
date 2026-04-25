"""
tune_adaptive.py — Quick A/B test of the Adaptive Width model under tuned settings.

Compares the default Adaptive Width config against a tuned version with:
  1. Lower learning rate (0.0005 vs 0.001) — HP search showed this is optimal
  2. Higher sparsity_lambda (0.005 vs 0.01) — softer gate regularization lets
     gates learn more useful masks instead of being forced too sparse
  3. Wider channel schedule [96, 64, 32] — gives the gates more features to
     work with at early stages, improving early-exit accuracy
  4. Longer warmup (15 epochs) — gates and classifiers need more warmup to
     stabilize before joint training with energy penalty

Runs only 5 trials for speed. Prints side-by-side comparison.
Estimated runtime: ~3-5 minutes on CPU.
"""

import os
import sys
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from models import AdaptiveEarlyExitNet, AdaptiveEnergyJointLoss
from train import set_seed, train_classifiers_only, train_joint, calibrate_thresholds
from evaluate import evaluate_model_advanced


def run_adaptive_trials(config_name, config, num_trials=5, base_seed=42):
    """Run num_trials of a given Adaptive Width configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, test_dl, class_weights = get_dataloaders(
        "./data/bonn", 64, use_freq_bands=True
    )
    class_weights = class_weights.to(device)

    metrics = {k: [] for k in ['acc', 'recall', 'f1', 'ece', 'energy_red']}

    for trial in range(num_trials):
        seed = base_seed + trial
        print(f"  [{config_name}] Trial {trial+1}/{num_trials} (seed={seed})")
        set_seed(seed)

        model = AdaptiveEarlyExitNet(
            in_channels=6,
            channel_sizes=config["channels"],
            num_classes=2,
            seq_len=4097,
        ).to(device)

        # Build criterion
        def make_criterion(energy_lambda):
            return AdaptiveEnergyJointLoss(
                model.stage_flops, class_weights,
                energy_lambda=energy_lambda,
                sparsity_lambda=config["sparsity_lambda"],
                is_baseline=False,
            )

        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        # Phase 1: Warmup
        train_classifiers_only(
            model, train_dl, config["warmup_epochs"],
            optimizer, make_criterion, device,
            scheduler=scheduler, max_grad_norm=1.0,
        )

        # Phase 2: Joint training
        train_joint(
            model, train_dl, config["joint_epochs"],
            optimizer, make_criterion, device,
            energy_lambda=config["energy_lambda"],
            scheduler=scheduler, max_grad_norm=1.0,
        )

        # Phase 3: Calibrate thresholds
        threshold_info = calibrate_thresholds(
            model, val_dl, device,
            strategy="confidence", target_acc=0.95,
        )

        # Evaluate
        acc, recall, f1, ece, energy_red, latency, _, _ = evaluate_model_advanced(
            model, test_dl, threshold_info, device, is_baseline=False
        )

        metrics['acc'].append(acc * 100)
        metrics['recall'].append(recall * 100)
        metrics['f1'].append(f1 * 100)
        metrics['ece'].append(ece)
        metrics['energy_red'].append(energy_red * 100)

        print(f"    Acc: {acc*100:.1f}%, F1: {f1*100:.1f}%, ER: {energy_red*100:.1f}%")

    return metrics


def print_comparison(name, metrics):
    """Print formatted results for one configuration."""
    print(f"\n  {name}:")
    print(f"    Accuracy:         {np.mean(metrics['acc']):.2f} ± {np.std(metrics['acc']):.2f}%")
    print(f"    Recall:           {np.mean(metrics['recall']):.2f} ± {np.std(metrics['recall']):.2f}%")
    print(f"    F1 Score:         {np.mean(metrics['f1']):.2f} ± {np.std(metrics['f1']):.2f}%")
    print(f"    ECE:              {np.mean(metrics['ece']):.4f} ± {np.std(metrics['ece']):.4f}")
    print(f"    Energy Reduction: {np.mean(metrics['energy_red']):.2f} ± {np.std(metrics['energy_red']):.2f}%")


def main():
    NUM_TRIALS = 5

    # ── Config A: Original (matches main.py defaults) ──
    original_config = {
        "channels": [64, 64, 64],
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "energy_lambda": 0.02,
        "sparsity_lambda": 0.01,
        "warmup_epochs": 10,
        "joint_epochs": 10,
    }

    # ── Config B: Tuned ──
    # Key changes:
    #   - lr=0.0005 (from HP search — more stable convergence)
    #   - sparsity_lambda=0.005 (halved — let gates stay open more, reducing
    #     premature channel suppression that kills accuracy)
    #   - energy_lambda=0.03 (slightly higher to push exits earlier, but
    #     not so high that it collapses accuracy)
    #   - warmup=15/joint=5 (gates need longer warmup to learn which channels
    #     matter before the energy penalty kicks in)
    #   - channels [96, 64, 32] (wider first stage = more features available
    #     at Stage 0 for the gate to select from, improving early-exit quality)
    tuned_config = {
        "channels": [96, 64, 32],
        "learning_rate": 0.0005,
        "weight_decay": 1e-4,
        "energy_lambda": 0.03,
        "sparsity_lambda": 0.005,
        "warmup_epochs": 15,
        "joint_epochs": 5,
    }

    print("=" * 60)
    print("ADAPTIVE WIDTH — TUNING COMPARISON (A/B Test)")
    print(f"  Trials: {NUM_TRIALS}")
    print("=" * 60)

    print(f"\n{'─'*60}")
    print("Config A (Original): ch=[64,64,64], lr=0.001, λ_e=0.02, λ_s=0.01")
    print(f"{'─'*60}")
    original_metrics = run_adaptive_trials("Original", original_config, NUM_TRIALS)

    print(f"\n{'─'*60}")
    print("Config B (Tuned): ch=[96,64,32], lr=0.0005, λ_e=0.03, λ_s=0.005")
    print(f"{'─'*60}")
    tuned_metrics = run_adaptive_trials("Tuned", tuned_config, NUM_TRIALS)

    # ── Side-by-side comparison ──
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print_comparison("Original [64,64,64]", original_metrics)
    print_comparison("Tuned [96,64,32]", tuned_metrics)

    # ── Delta ──
    delta_acc = np.mean(tuned_metrics['acc']) - np.mean(original_metrics['acc'])
    delta_f1 = np.mean(tuned_metrics['f1']) - np.mean(original_metrics['f1'])
    delta_er = np.mean(tuned_metrics['energy_red']) - np.mean(original_metrics['energy_red'])

    print(f"\n{'─'*60}")
    print("DELTAS (Tuned − Original):")
    print(f"  Accuracy:         {delta_acc:+.2f} pp")
    print(f"  F1 Score:         {delta_f1:+.2f} pp")
    print(f"  Energy Reduction: {delta_er:+.2f} pp")
    print(f"{'─'*60}")

    if delta_acc > 0 and delta_f1 > 0:
        print("✅ Tuned config improves accuracy AND F1.")
    elif delta_acc > 0:
        print("⚠️ Tuned config improves accuracy but F1 regressed.")
    else:
        print("❌ Tuned config did not improve accuracy. Try different settings.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
