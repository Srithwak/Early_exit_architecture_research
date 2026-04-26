"""
tuning.py — Systematic hyperparameter search for early-exit networks.

Provides grid search and random search over key hyperparameters,
ensuring fair comparison by using identical seeds and evaluation protocols.
"""

import os
import itertools
import csv
import numpy as np
import torch
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List, Dict, Any

from train import set_seed, train_classifiers_only, train_joint, calibrate_thresholds
from evaluate import evaluate_model_advanced


@dataclass
class HyperparameterSearchSpace:
    """Defines the grid of hyperparameters to search over."""
    learning_rates: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3])
    energy_lambdas: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.1, 0.2])
    weight_decays: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    warmup_epochs_list: List[int] = field(default_factory=lambda: [5, 10, 15])
    joint_epochs_list: List[int] = field(default_factory=lambda: [5, 10, 15])
    sparsity_lambdas: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02])

    def total_combinations(self, include_sparsity=False):
        """Total number of hyperparameter combinations."""
        count = (len(self.learning_rates) * len(self.energy_lambdas) *
                 len(self.weight_decays) * len(self.warmup_epochs_list) *
                 len(self.joint_epochs_list))
        if include_sparsity:
            count *= len(self.sparsity_lambdas)
        return count


def run_hyperparameter_search(model_builder_fn, criterion_builder_fn, train_dl, val_dl, test_dl,
                              search_space: HyperparameterSearchSpace, device,
                              is_adaptive=False, is_baseline=False,
                              max_combinations=50, seed=42,
                              threshold_strategy="confidence",
                              results_dir="./results"):
    """
    Run a systematic hyperparameter search.
    
    Args:
        model_builder_fn: callable() -> nn.Module (creates a fresh model)
        criterion_builder_fn: callable(model, energy_lambda, sparsity_lambda) -> loss_fn
        train_dl, val_dl, test_dl: DataLoaders
        search_space: HyperparameterSearchSpace
        device: torch device
        is_adaptive: whether model uses channel gates (affects sparsity search)
        is_baseline: whether this is a baseline model
        max_combinations: maximum number of combinations to try (random subset if exceeded)
        seed: base random seed
        threshold_strategy: calibration strategy
        results_dir: where to save CSV results
    
    Returns:
        list of result dicts, sorted by F1 score (descending)
    """
    os.makedirs(results_dir, exist_ok=True)

    # Build the parameter grid
    if is_adaptive:
        param_grid = list(itertools.product(
            search_space.learning_rates,
            search_space.energy_lambdas,
            search_space.weight_decays,
            search_space.warmup_epochs_list,
            search_space.joint_epochs_list,
            search_space.sparsity_lambdas
        ))
        param_keys = ["lr", "energy_lambda", "weight_decay", "warmup_epochs", "joint_epochs", "sparsity_lambda"]
    else:
        param_grid = list(itertools.product(
            search_space.learning_rates,
            search_space.energy_lambdas,
            search_space.weight_decays,
            search_space.warmup_epochs_list,
            search_space.joint_epochs_list,
        ))
        param_keys = ["lr", "energy_lambda", "weight_decay", "warmup_epochs", "joint_epochs"]

    # Random subset if too many combinations
    if len(param_grid) > max_combinations:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(param_grid), max_combinations, replace=False)
        param_grid = [param_grid[i] for i in indices]
        print(f"[Tuning] Randomly sampling {max_combinations} of {len(param_grid)} total combinations")

    print(f"[Tuning] Running {len(param_grid)} hyperparameter combinations")

    results = []

    for idx, params in enumerate(param_grid):
        param_dict = dict(zip(param_keys, params))
        print(f"\n[Tuning {idx+1}/{len(param_grid)}] {param_dict}")

        set_seed(seed)
        model = model_builder_fn().to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=param_dict["lr"],
            weight_decay=param_dict["weight_decay"]
        )

        e_lambda = param_dict["energy_lambda"]
        s_lambda = param_dict.get("sparsity_lambda", 0.01)

        def criterion_fn(energy_lambda):
            return criterion_builder_fn(model, energy_lambda, s_lambda)

        # Training
        try:
            train_classifiers_only(
                model, train_dl, param_dict["warmup_epochs"],
                optimizer, criterion_fn, device
            )

            if not is_baseline:
                train_joint(
                    model, train_dl, param_dict["joint_epochs"],
                    optimizer, criterion_fn, device,
                    energy_lambda=e_lambda
                )

            # Calibration
            threshold_info = calibrate_thresholds(
                model, val_dl, device, strategy=threshold_strategy
            )

            # Evaluation
            acc, recall, f1, ece, energy_red, latency, _, _ = evaluate_model_advanced(
                model, test_dl, threshold_info, device, is_baseline=is_baseline
            )

            result = {
                **param_dict,
                "accuracy": acc * 100,
                "recall": recall * 100,
                "f1": f1 * 100,
                "ece": ece,
                "energy_reduction": energy_red * 100,
                "latency_ms": latency,
                "status": "success"
            }

        except Exception as e:
            print(f"  [ERROR] {e}")
            result = {
                **param_dict,
                "accuracy": 0.0, "recall": 0.0, "f1": 0.0,
                "ece": 1.0, "energy_reduction": 0.0, "latency_ms": 0.0,
                "status": f"error: {str(e)}"
            }

        results.append(result)
        print(f"  -> Acc: {result['accuracy']:.2f}%, F1: {result['f1']:.2f}%, "
              f"Energy Red: {result['energy_reduction']:.2f}%")

    # Sort by F1 score
    results.sort(key=lambda r: r["f1"], reverse=True)

    # Save to CSV
    csv_path = os.path.join(results_dir, "hyperparameter_search.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Tuning] Results saved to {csv_path}")

    # Print top 5
    print(f"\n{'='*60}")
    print("Top 5 Configurations:")
    print(f"{'='*60}")
    for i, r in enumerate(results[:5]):
        params_str = ", ".join(f"{k}={v}" for k, v in r.items()
                               if k not in ["accuracy", "recall", "f1", "ece",
                                             "energy_reduction", "latency_ms", "status"])
        print(f"  #{i+1}: F1={r['f1']:.2f}%, Acc={r['accuracy']:.2f}%, "
              f"ER={r['energy_reduction']:.2f}% | {params_str}")

    return results
