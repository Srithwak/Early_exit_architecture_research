import os
import itertools
import csv
import numpy as np
import torch
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List

from train import set_seed, train_classifiers_only, train_joint, calibrate_thresholds
from evaluate import evaluate_model_advanced


@dataclass
class HyperparameterSearchSpace:
    learning_rates: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3])
    energy_lambdas: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.1, 0.2])
    weight_decays: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    warmup_epochs_list: List[int] = field(default_factory=lambda: [5, 10, 15])
    joint_epochs_list: List[int] = field(default_factory=lambda: [5, 10, 15])
    sparsity_lambdas: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02])


def run_hyperparameter_search(model_builder_fn, criterion_builder_fn, train_dl, val_dl, test_dl,
                              search_space, device, is_adaptive=False, is_baseline=False,
                              max_combinations=50, seed=42, threshold_strategy="confidence",
                              results_dir="./results"):
    os.makedirs(results_dir, exist_ok=True)

    if is_adaptive:
        param_grid = list(itertools.product(
            search_space.learning_rates, search_space.energy_lambdas,
            search_space.weight_decays, search_space.warmup_epochs_list,
            search_space.joint_epochs_list, search_space.sparsity_lambdas))
        param_keys = ["lr", "energy_lambda", "weight_decay", "warmup_epochs", "joint_epochs", "sparsity_lambda"]
    else:
        param_grid = list(itertools.product(
            search_space.learning_rates, search_space.energy_lambdas,
            search_space.weight_decays, search_space.warmup_epochs_list,
            search_space.joint_epochs_list))
        param_keys = ["lr", "energy_lambda", "weight_decay", "warmup_epochs", "joint_epochs"]

    if len(param_grid) > max_combinations:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(param_grid), max_combinations, replace=False)
        param_grid = [param_grid[i] for i in indices]

    print(f"[Tuning] {len(param_grid)} combinations")
    results = []

    for idx, params in enumerate(param_grid):
        pd = dict(zip(param_keys, params))
        print(f"\n  [{idx+1}/{len(param_grid)}] {pd}")

        set_seed(seed)
        model = model_builder_fn().to(device)
        optimizer = optim.Adam(model.parameters(), lr=pd["lr"], weight_decay=pd["weight_decay"])

        e_lambda = pd["energy_lambda"]
        s_lambda = pd.get("sparsity_lambda", 0.01)

        def criterion_fn(energy_lambda):
            return criterion_builder_fn(model, energy_lambda, s_lambda)

        try:
            train_classifiers_only(model, train_dl, pd["warmup_epochs"], optimizer, criterion_fn, device)
            if not is_baseline:
                train_joint(model, train_dl, pd["joint_epochs"], optimizer, criterion_fn, device, energy_lambda=e_lambda)

            threshold_info = calibrate_thresholds(model, val_dl, device, strategy=threshold_strategy)
            acc, recall, f1, ece, energy_red, latency, _, _ = evaluate_model_advanced(
                model, test_dl, threshold_info, device, is_baseline=is_baseline)

            result = {**pd, "accuracy": acc * 100, "recall": recall * 100, "f1": f1 * 100,
                      "ece": ece, "energy_reduction": energy_red * 100, "latency_ms": latency, "status": "success"}
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {**pd, "accuracy": 0.0, "recall": 0.0, "f1": 0.0,
                      "ece": 1.0, "energy_reduction": 0.0, "latency_ms": 0.0, "status": f"error: {e}"}

        results.append(result)
        print(f"  -> Acc: {result['accuracy']:.2f}%, F1: {result['f1']:.2f}%")

    results.sort(key=lambda r: r["f1"], reverse=True)

    csv_path = os.path.join(results_dir, "hyperparameter_search.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nTop 3:")
    for i, r in enumerate(results[:3]):
        print(f"  #{i+1}: F1={r['f1']:.2f}%, Acc={r['accuracy']:.2f}%")

    return results
