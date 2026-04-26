"""
main_mitbih.py — Early Exit Architecture Research Pipeline for MIT-BIH Dataset.

Mirrors the Bonn pipeline (main.py) but configured for the MIT-BIH
Arrhythmia dataset (5-class, 187 seq_len, 125 Hz).

All model configurations are tested under identical conditions
(same seeds, same total epochs, same evaluation protocol) to
ensure unbiased comparison.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List

from dataset_mitbih import get_mitbih_dataloaders
from models import EnergyJointLoss, AdaptiveEnergyJointLoss
from models_mitbih import (GenericEarlyExitNetSmall, AdaptiveEarlyExitNetSmall,
                           apply_structured_pruning_small)
from train import set_seed, train_classifiers_only, train_joint, calibrate_thresholds
from evaluate import inspect_data, evaluate_model_advanced
from analysis import (collect_exit_statistics, analyze_exit_patterns,
                      compute_difficulty_scores, print_analysis_report)
from visualize import generate_all_plots
from statistical_tests import (compute_all_statistics, print_statistical_report,
                               confidence_interval)


# ──────────────────────────────────────────────
# Configuration — MIT-BIH specific
# ──────────────────────────────────────────────

@dataclass
class MITBIHExperimentConfig:
    """Centralized configuration for MIT-BIH experiments."""
    data_dir: str = "./data/mitbih"
    plots_dir: str = "./plots_mitbih"
    results_dir: str = "./results_mitbih"
    batch_size: int = 128            # Larger batches since dataset is much bigger
    seq_len: int = 187               # MIT-BIH heartbeat length
    use_freq_bands: bool = True
    in_channels: int = 6             # 1 Raw Time-Series + 5 Frequency Bands
    num_classes: int = 5             # N, S, V, F, Q arrhythmia classes
    num_trials: int = 1              # Multiple trials for statistical validity
    base_seed: int = 42
    # Training — same total epochs for ALL models (unbiased)
    # Reduced for CPU feasibility with 76K samples (~3 min/epoch)
    warmup_epochs: int = 1
    joint_epochs: int = 1
    # Hyperparameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    energy_lambda: float = 0.02
    sparsity_lambda: float = 0.01
    max_grad_norm: float = 1.0
    # Threshold strategy
    threshold_strategy: str = "confidence"  # "confidence", "entropy", "patience"
    target_acc: float = 0.90         # Slightly lower target for 5-class
    entropy_percentile: float = 80
    patience: int = 2


# ──────────────────────────────────────────────
# Model Registry (same architectures, adapted for MIT-BIH)
# ──────────────────────────────────────────────

def get_architecture_configs():
    """
    All model architectures to test. Every non-baseline model
    uses identical training protocol (same warmup + joint epochs).
    """
    return {
        "Base CNN (Control)": {
            "channels": [64, 64, 64],
            "is_baseline": True,
            "adaptive": False,
            "description": "Standard CNN without early exits (control group)"
        },
        "Constant Width": {
            "channels": [64, 64, 64],
            "is_baseline": False,
            "adaptive": False,
            "description": "Equal channel count across stages"
        },
        "Increasing Width": {
            "channels": [32, 64, 128],
            "is_baseline": False,
            "adaptive": False,
            "description": "Channels increase with depth"
        },
        "Decreasing Width": {
            "channels": [128, 64, 32],
            "is_baseline": False,
            "adaptive": False,
            "description": "Channels decrease with depth"
        },
        "Adaptive Width": {
            "channels": [64, 64, 64],
            "is_baseline": False,
            "adaptive": True,
            "description": "Learned channel gating (soft pruning)"
        },
    }


def get_model_size_configs():
    """Model size variants for scaling experiments."""
    return {
        "Tiny [16]": {"channels": [16, 16, 16]},
        "Small [32]": {"channels": [32, 32, 32]},
        "Medium [64]": {"channels": [64, 64, 64]},
        "Large [128]": {"channels": [128, 128, 128]},
        "XLarge [256]": {"channels": [256, 256, 256]},
    }


# ──────────────────────────────────────────────
# Research Pipeline (MIT-BIH)
# ──────────────────────────────────────────────

class MITBIHResearchPipeline:
    """Pipeline for MIT-BIH Early-Exit experiments, mirroring the Bonn pipeline."""

    def __init__(self, config: MITBIHExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MIT-BIH Pipeline] Device: {self.device}")
        print(f"[MIT-BIH Pipeline] Dataset: MIT-BIH Arrhythmia")
        print(f"[MIT-BIH Pipeline] Classes: {config.num_classes}, Seq Len: {config.seq_len}")

        os.makedirs(config.plots_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)

        self.train_dl, self.val_dl, self.test_dl, self.class_weights = get_mitbih_dataloaders(
            config.data_dir, config.batch_size, use_freq_bands=config.use_freq_bands
        )
        self.class_weights = self.class_weights.to(self.device)
        self.results = []
        self.all_trial_metrics = {}
        self.viz_data = {"per_model": {}, "exit_distributions": {}}

    def _build_model(self, config_dict):
        """Factory method to construct the appropriate network (small blocks for MIT-BIH)."""
        if config_dict.get("adaptive"):
            return AdaptiveEarlyExitNetSmall(
                in_channels=self.config.in_channels,
                channel_sizes=config_dict["channels"],
                num_classes=self.config.num_classes,
                seq_len=self.config.seq_len
            ).to(self.device)
        return GenericEarlyExitNetSmall(
            in_channels=self.config.in_channels,
            channel_sizes=config_dict["channels"],
            num_classes=self.config.num_classes,
            seq_len=self.config.seq_len
        ).to(self.device)

    def _get_criterion_fn(self, model, config_dict):
        """Returns a closure that builds the loss for a given energy_lambda."""
        def criterion_fn(energy_lambda):
            if config_dict.get("adaptive"):
                return AdaptiveEnergyJointLoss(
                    model.stage_flops, self.class_weights,
                    energy_lambda=energy_lambda,
                    sparsity_lambda=self.config.sparsity_lambda,
                    is_baseline=config_dict["is_baseline"]
                )
            return EnergyJointLoss(
                model.stage_flops, self.class_weights,
                energy_lambda=energy_lambda,
                is_baseline=config_dict["is_baseline"]
            )
        return criterion_fn

    def _train_model(self, model, config_dict, seed):
        """
        Train a model using the two-phase protocol.

        UNBIASED: All models (including baseline) get the same total
        number of epochs. Baselines train warmup_epochs + joint_epochs
        all in warmup mode. Non-baselines split into warmup then joint.
        """
        set_seed(seed)
        criterion_fn = self._get_criterion_fn(model, config_dict)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        total_epochs = self.config.warmup_epochs + self.config.joint_epochs

        if config_dict["is_baseline"]:
            # Baseline: all epochs in warmup (no exit policy training)
            train_classifiers_only(
                model, self.train_dl, total_epochs,
                optimizer, criterion_fn, self.device,
                scheduler=scheduler, max_grad_norm=self.config.max_grad_norm
            )
            threshold_info = {"strategy": "confidence",
                              "thresholds": [1.0] * (model.num_stages - 1)}
        else:
            # Non-baseline: warmup then joint
            train_classifiers_only(
                model, self.train_dl, self.config.warmup_epochs,
                optimizer, criterion_fn, self.device,
                scheduler=scheduler, max_grad_norm=self.config.max_grad_norm
            )
            train_joint(
                model, self.train_dl, self.config.joint_epochs,
                optimizer, criterion_fn, self.device,
                energy_lambda=self.config.energy_lambda,
                scheduler=scheduler, max_grad_norm=self.config.max_grad_norm
            )
            threshold_info = calibrate_thresholds(
                model, self.val_dl, self.device,
                strategy=self.config.threshold_strategy,
                target_acc=self.config.target_acc,
                entropy_percentile=self.config.entropy_percentile,
                patience=self.config.patience
            )

        return threshold_info

    def run_architecture_ablation(self, energy_lambdas=None):
        """Run experiments across all architectures with multi-trial averaging."""
        arch_configs = get_architecture_configs()
        if energy_lambdas is None:
            energy_lambdas = [self.config.energy_lambda]

        for e_lambda in energy_lambdas:
            print(f"\n{'='*60}")
            print(f"[MIT-BIH] Energy Lambda = {e_lambda}")
            print(f"{'='*60}")

            for name, m_config in arch_configs.items():
                print(f"\n--- {name} ({self.config.num_trials} trials) ---")
                trial_metrics = {k: [] for k in
                    ['acc', 'recall', 'f1', 'ece', 'energy_red', 'latency']}
                last_exit_df = None
                last_analysis = None
                last_ece_bin_data = None

                for trial in range(self.config.num_trials):
                    trial_seed = self.config.base_seed + trial
                    print(f"  Trial {trial+1}/{self.config.num_trials} (seed={trial_seed})")

                    self.config.energy_lambda = e_lambda
                    model = self._build_model(m_config)
                    total_params = model.count_parameters()
                    total_flops = sum(model.stage_flops) / 1e6

                    threshold_info = self._train_model(model, m_config, trial_seed)

                    # Evaluate
                    acc, recall, f1, ece, energy_red, latency, per_sample, ece_bins = \
                        evaluate_model_advanced(model, self.test_dl, threshold_info,
                                                self.device, is_baseline=m_config["is_baseline"])

                    trial_metrics['acc'].append(acc * 100)
                    trial_metrics['recall'].append(recall * 100)
                    trial_metrics['f1'].append(f1 * 100)
                    trial_metrics['ece'].append(ece)
                    trial_metrics['energy_red'].append(energy_red * 100)
                    trial_metrics['latency'].append(latency)

                    # Deep analysis on last trial
                    if trial == self.config.num_trials - 1:
                        exit_df = collect_exit_statistics(
                            model, self.test_dl, threshold_info,
                            self.device, is_baseline=m_config["is_baseline"])
                        last_analysis = analyze_exit_patterns(exit_df, model.num_stages)
                        last_exit_df = exit_df
                        last_ece_bin_data = ece_bins
                        print_analysis_report(last_analysis, name)

                    print(f"    Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%, "
                          f"ER: {energy_red*100:.2f}%")

                # Store raw trial metrics for statistical testing
                self.all_trial_metrics[name] = trial_metrics

                # Aggregate
                result = {
                    "Model": name,
                    "Lambda": e_lambda,
                    "Params": total_params,
                    "MFLOPs": f"{total_flops:.2f}",
                    "Accuracy (%)": f"{np.mean(trial_metrics['acc']):.2f} +/- {np.std(trial_metrics['acc']):.2f}",
                    "Recall (%)": f"{np.mean(trial_metrics['recall']):.2f} +/- {np.std(trial_metrics['recall']):.2f}",
                    "F1 Score (%)": f"{np.mean(trial_metrics['f1']):.2f} +/- {np.std(trial_metrics['f1']):.2f}",
                    "ECE": f"{np.mean(trial_metrics['ece']):.4f} +/- {np.std(trial_metrics['ece']):.4f}",
                    "Energy Red (%)": f"{np.mean(trial_metrics['energy_red']):.2f} +/- {np.std(trial_metrics['energy_red']):.2f}",
                    "Latency (ms)": f"{np.mean(trial_metrics['latency']):.2f} +/- {np.std(trial_metrics['latency']):.2f}",
                }
                self.results.append(result)

                # Store viz data
                stage_exits = [0] * model.num_stages
                if last_exit_df is not None:
                    for s in range(model.num_stages):
                        stage_exits[s] = int((last_exit_df["exit_stage"] == s).sum())
                self.viz_data["exit_distributions"][name] = stage_exits
                self.viz_data["per_model"][name] = {
                    "ece_bin_data": last_ece_bin_data,
                    "ece": np.mean(trial_metrics['ece']),
                    "analysis": last_analysis,
                    "exit_df": last_exit_df,
                }

        # Run statistical significance tests
        if self.all_trial_metrics and self.config.num_trials >= 2:
            print("\n--- Running Statistical Significance Tests ---")
            model_names = list(self.all_trial_metrics.keys())
            stats_results = compute_all_statistics(self.all_trial_metrics, model_names)
            print_statistical_report(stats_results)
            self.viz_data["statistical_tests"] = stats_results

        df = pd.DataFrame(self.results)
        self.viz_data["results_df"] = df
        self.viz_data["results_list"] = self.results
        return df

    def run_model_size_experiment(self):
        """Experiment with different model sizes."""
        size_configs = get_model_size_configs()
        size_results = []

        print(f"\n{'='*60}")
        print("[MIT-BIH] Model Size Scaling Experiment")
        print(f"{'='*60}")

        for name, s_config in size_configs.items():
            m_config = {"channels": s_config["channels"], "is_baseline": False, "adaptive": False}
            model = self._build_model(m_config)
            params = model.count_parameters()
            flops = sum(model.stage_flops) / 1e6
            print(f"\n--- {name}: {params:,} params, {flops:.2f} MFLOPs ---")

            set_seed(self.config.base_seed)
            threshold_info = self._train_model(model, m_config, self.config.base_seed)

            acc, recall, f1, ece, energy_red, latency, _, _ = \
                evaluate_model_advanced(model, self.test_dl, threshold_info,
                                        self.device, is_baseline=False)

            size_results.append({
                "name": name, "channels": s_config["channels"],
                "params": params, "mflops": flops,
                "accuracy": acc * 100, "f1": f1 * 100,
                "energy_reduction": energy_red * 100,
            })
            print(f"    Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%, ER: {energy_red*100:.2f}%")

        self.viz_data["size_results"] = size_results
        return pd.DataFrame(size_results)

    def run_pruning_experiment(self, prune_ratios=None):
        """Experiment with structured pruning at different ratios."""
        if prune_ratios is None:
            prune_ratios = [0.0, 0.1, 0.25, 0.5]

        pruning_results = []
        print(f"\n{'='*60}")
        print("[MIT-BIH] Structured Pruning Experiment")
        print(f"{'='*60}")

        m_config = {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False}

        # Train base model first
        set_seed(self.config.base_seed)
        base_model = self._build_model(m_config)
        self._train_model(base_model, m_config, self.config.base_seed)
        base_flops = sum(base_model.stage_flops)

        for ratio in prune_ratios:
            print(f"\n--- Prune Ratio: {ratio:.0%} ---")
            if ratio == 0.0:
                model = base_model
                pruned_flops = base_flops
            else:
                model, pruned_ch = apply_structured_pruning_small(
                    base_model, prune_ratio=ratio,
                    in_channels=self.config.in_channels,
                    seq_len=self.config.seq_len,
                    num_classes=self.config.num_classes
                )
                model = model.to(self.device)
                pruned_flops = sum(model.stage_flops)

                # Fine-tune the pruned model
                criterion_fn = self._get_criterion_fn(model, m_config)
                optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.1)
                train_joint(model, self.train_dl, 5, optimizer, criterion_fn,
                            self.device, energy_lambda=self.config.energy_lambda)

            threshold_info = calibrate_thresholds(
                model, self.val_dl, self.device,
                strategy=self.config.threshold_strategy
            )
            acc, recall, f1, ece, energy_red, latency, _, _ = \
                evaluate_model_advanced(model, self.test_dl, threshold_info,
                                        self.device, is_baseline=False)

            flops_red = (1.0 - pruned_flops / base_flops) * 100
            pruning_results.append({
                "prune_ratio": ratio,
                "params": model.count_parameters(),
                "accuracy": acc * 100, "f1": f1 * 100,
                "flops_reduction": flops_red,
                "energy_reduction": energy_red * 100,
            })
            print(f"    Acc: {acc*100:.2f}%, FLOPs Red: {flops_red:.1f}%")

        self.viz_data["pruning_results"] = pruning_results
        return pd.DataFrame(pruning_results)

    def run_threshold_strategy_comparison(self):
        """Compare different threshold strategies on the same model."""
        strategies = ["confidence", "entropy", "patience"]
        strategy_results = {}

        print(f"\n{'='*60}")
        print("[MIT-BIH] Threshold Strategy Comparison")
        print(f"{'='*60}")

        m_config = {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False}
        set_seed(self.config.base_seed)
        model = self._build_model(m_config)
        self._train_model(model, m_config, self.config.base_seed)

        for strategy in strategies:
            print(f"\n--- Strategy: {strategy} ---")
            threshold_info = calibrate_thresholds(
                model, self.val_dl, self.device, strategy=strategy,
                target_acc=self.config.target_acc,
                entropy_percentile=self.config.entropy_percentile,
                patience=self.config.patience
            )
            acc, recall, f1, ece, energy_red, latency, _, _ = \
                evaluate_model_advanced(model, self.test_dl, threshold_info,
                                        self.device, is_baseline=False)

            strategy_results[strategy] = {
                "accuracy": acc * 100, "f1": f1 * 100,
                "energy_reduction": energy_red * 100,
                "ece": ece, "latency_ms": latency,
            }
            print(f"    Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%, ER: {energy_red*100:.2f}%")

        self.viz_data["strategy_results"] = strategy_results
        return strategy_results

    def generate_visualizations(self):
        """Generate all plots from collected data."""
        generate_all_plots(self.viz_data, self.config.plots_dir)

    def save_results(self):
        """Save all raw results to disk for reproducibility."""
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = os.path.join(self.config.results_dir, "main_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")

        # Save comprehensive JSON with raw trial data
        json_output = {
            "dataset": "MIT-BIH Arrhythmia",
            "config": {
                "batch_size": self.config.batch_size,
                "seq_len": self.config.seq_len,
                "in_channels": self.config.in_channels,
                "num_classes": self.config.num_classes,
                "num_trials": self.config.num_trials,
                "base_seed": self.config.base_seed,
                "warmup_epochs": self.config.warmup_epochs,
                "joint_epochs": self.config.joint_epochs,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "energy_lambda": self.config.energy_lambda,
                "threshold_strategy": self.config.threshold_strategy,
            },
            "results_summary": self.results,
            "raw_trial_metrics": {
                name: {k: [float(v) for v in vals] for k, vals in metrics.items()}
                for name, metrics in self.all_trial_metrics.items()
            },
        }

        # Add statistical test results if available
        if "statistical_tests" in self.viz_data:
            stats = self.viz_data["statistical_tests"]
            json_stats = {}
            for key in ["confidence_intervals"]:
                if key in stats:
                    json_stats[key] = stats[key]
            if "pairwise_comparisons" in stats:
                json_stats["pairwise_comparisons"] = {}
                for metric, comparisons in stats["pairwise_comparisons"].items():
                    json_stats["pairwise_comparisons"][metric] = [
                        {k: v for k, v in c.items()} for c in comparisons
                    ]
            json_output["statistical_analysis"] = json_stats

        json_path = os.path.join(self.config.results_dir, "experiment_results.json")
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2, default=str)
        print(f"Comprehensive results saved to {json_path}")


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Early Exit Architecture Research — MIT-BIH Dataset")
    parser.add_argument("--run-ablation", action="store_true",
                        help="Run architecture ablation study")
    parser.add_argument("--run-sizes", action="store_true",
                        help="Run model size scaling experiment")
    parser.add_argument("--run-pruning", action="store_true",
                        help="Run structured pruning experiment")
    parser.add_argument("--run-strategies", action="store_true",
                        help="Compare threshold strategies")
    parser.add_argument("--run-tuning", action="store_true",
                        help="Run hyperparameter search")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all experiments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per configuration")
    parser.add_argument("--strategy", type=str, default="confidence",
                        choices=["confidence", "entropy", "patience"],
                        help="Threshold calibration strategy")
    args = parser.parse_args()

    cfg = MITBIHExperimentConfig(
        base_seed=args.seed,
        num_trials=args.trials,
        threshold_strategy=args.strategy,
    )
    pipeline = MITBIHResearchPipeline(cfg)
    inspect_data(pipeline.train_dl)

    run_any = args.run_ablation or args.run_sizes or args.run_pruning or \
              args.run_strategies or args.run_tuning or args.run_all

    if not run_any:
        # Default: run architecture ablation
        args.run_ablation = True

    if args.run_all or args.run_ablation:
        print("\n" + "="*60)
        print("ARCHITECTURE ABLATION STUDY — MIT-BIH")
        print("="*60)
        df = pipeline.run_architecture_ablation()
        print("\n" + df.to_string())

    if args.run_all or args.run_strategies:
        print("\n" + "="*60)
        print("THRESHOLD STRATEGY COMPARISON — MIT-BIH")
        print("="*60)
        pipeline.run_threshold_strategy_comparison()

    if args.run_all or args.run_sizes:
        print("\n" + "="*60)
        print("MODEL SIZE SCALING — MIT-BIH")
        print("="*60)
        df_sizes = pipeline.run_model_size_experiment()
        print("\n" + df_sizes.to_string())

    if args.run_all or args.run_pruning:
        print("\n" + "="*60)
        print("STRUCTURED PRUNING — MIT-BIH")
        print("="*60)
        df_pruning = pipeline.run_pruning_experiment()
        print("\n" + df_pruning.to_string())

    if args.run_all or args.run_tuning:
        print("\n" + "="*60)
        print("HYPERPARAMETER SEARCH — MIT-BIH")
        print("="*60)
        from tuning import HyperparameterSearchSpace, run_hyperparameter_search

        m_config = {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False}
        search_space = HyperparameterSearchSpace(
            learning_rates=[5e-4, 1e-3, 5e-3],
            energy_lambdas=[0.01, 0.05, 0.1],
            weight_decays=[1e-4, 1e-3],
            warmup_epochs_list=[8, 12],
            joint_epochs_list=[8, 12],
        )
        tuning_results = run_hyperparameter_search(
            model_builder_fn=lambda: pipeline._build_model(m_config),
            criterion_builder_fn=lambda model, el, sl: EnergyJointLoss(
                model.stage_flops, pipeline.class_weights, energy_lambda=el),
            train_dl=pipeline.train_dl, val_dl=pipeline.val_dl,
            test_dl=pipeline.test_dl, search_space=search_space,
            device=pipeline.device, max_combinations=30, seed=args.seed,
            results_dir=cfg.results_dir,
        )
        pipeline.viz_data["tuning_results"] = tuning_results

    # Generate all visualizations and save results
    pipeline.generate_visualizations()
    pipeline.save_results()
    print("\n[DONE] All MIT-BIH experiments complete.")
