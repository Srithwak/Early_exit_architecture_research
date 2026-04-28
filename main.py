import os, json, torch, argparse
import pandas as pd
import numpy as np
import torch.optim as optim
from dataclasses import dataclass

from dataset import get_dataloaders
from models import (GenericEarlyExitNet, AdaptiveEarlyExitNet,
                    EnergyJointLoss, AdaptiveEnergyJointLoss, apply_structured_pruning)
from train import set_seed, train_classifiers_only, train_joint, calibrate_thresholds
from evaluate import inspect_data, evaluate_model_advanced
from analysis import collect_exit_statistics, analyze_exit_patterns, print_analysis_report
from visualize import generate_all_plots
from statistical_tests import compute_all_statistics, print_statistical_report


@dataclass
class Config:
    data_dir: str = "./data/bonn"
    plots_dir: str = "./plots"
    results_dir: str = "./results"
    batch_size: int = 64
    seq_len: int = 4097
    in_channels: int = 6
    num_classes: int = 2
    num_trials: int = 5
    base_seed: int = 42
    warmup_epochs: int = 10
    joint_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    energy_lambda: float = 0.02
    sparsity_lambda: float = 0.01
    max_grad_norm: float = 1.0
    threshold_strategy: str = "confidence"
    target_acc: float = 0.95
    entropy_percentile: float = 80
    patience: int = 2


ARCHITECTURES = {
    "Base CNN (Control)":  {"channels": [64, 64, 64], "is_baseline": True, "adaptive": False},
    "Constant Width":      {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False},
    "Increasing Width":    {"channels": [32, 64, 128], "is_baseline": False, "adaptive": False},
    "Decreasing Width":    {"channels": [128, 64, 32], "is_baseline": False, "adaptive": False},
    "Adaptive Width":      {"channels": [64, 64, 64], "is_baseline": False, "adaptive": True},
}

SIZE_CONFIGS = {
    "Tiny [16]": [16, 16, 16], "Small [32]": [32, 32, 32], "Medium [64]": [64, 64, 64],
    "Large [128]": [128, 128, 128], "XLarge [256]": [256, 256, 256],
}


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cpu")
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        print(f"Device: {self.device} ({torch.get_num_threads()} threads)")
        os.makedirs(cfg.plots_dir, exist_ok=True)
        os.makedirs(cfg.results_dir, exist_ok=True)

        self.train_dl, self.val_dl, self.test_dl, self.class_weights = get_dataloaders(
            cfg.data_dir, cfg.batch_size, use_freq_bands=True)
        self.class_weights = self.class_weights.to(self.device)
        self.results = []
        self.all_trial_metrics = {}
        self.viz_data = {"per_model": {}, "exit_distributions": {}}

    def _build(self, mc):
        if mc.get("adaptive"):
            return AdaptiveEarlyExitNet(self.cfg.in_channels, mc["channels"],
                                        self.cfg.num_classes, self.cfg.seq_len).to(self.device)
        return GenericEarlyExitNet(self.cfg.in_channels, mc["channels"],
                                   self.cfg.num_classes, self.cfg.seq_len).to(self.device)

    def _criterion_fn(self, model, mc):
        def fn(energy_lambda):
            if mc.get("adaptive"):
                return AdaptiveEnergyJointLoss(model.stage_flops, self.class_weights,
                    energy_lambda=energy_lambda, sparsity_lambda=self.cfg.sparsity_lambda, is_baseline=mc["is_baseline"])
            return EnergyJointLoss(model.stage_flops, self.class_weights,
                energy_lambda=energy_lambda, is_baseline=mc["is_baseline"])
        return fn

    def _train(self, model, mc, seed):
        set_seed(seed)
        criterion_fn = self._criterion_fn(model, mc)
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        total = self.cfg.warmup_epochs + self.cfg.joint_epochs

        if mc["is_baseline"]:
            train_classifiers_only(model, self.train_dl, total, optimizer, criterion_fn, self.device,
                                   scheduler=scheduler, max_grad_norm=self.cfg.max_grad_norm)
            return {"strategy": "confidence", "thresholds": [1.0] * (model.num_stages - 1)}
        else:
            train_classifiers_only(model, self.train_dl, self.cfg.warmup_epochs, optimizer, criterion_fn, self.device,
                                   scheduler=scheduler, max_grad_norm=self.cfg.max_grad_norm)
            train_joint(model, self.train_dl, self.cfg.joint_epochs, optimizer, criterion_fn, self.device,
                        energy_lambda=self.cfg.energy_lambda, scheduler=scheduler, max_grad_norm=self.cfg.max_grad_norm)
            return calibrate_thresholds(model, self.val_dl, self.device, strategy=self.cfg.threshold_strategy,
                                        target_acc=self.cfg.target_acc, entropy_percentile=self.cfg.entropy_percentile,
                                        patience=self.cfg.patience)

    def run_ablation(self):
        for name, mc in ARCHITECTURES.items():
            print(f"\n--- {name} ({self.cfg.num_trials} trials) ---")
            trial_metrics = {k: [] for k in ['acc', 'recall', 'f1', 'ece', 'energy_red', 'latency']}
            last_exit_df = last_analysis = last_ece_bins = None

            for trial in range(self.cfg.num_trials):
                seed = self.cfg.base_seed + trial
                model = self._build(mc)
                total_params = model.count_parameters()
                total_flops = sum(model.stage_flops) / 1e6

                threshold_info = self._train(model, mc, seed)
                acc, recall, f1, ece, er, lat, _, ece_bins = evaluate_model_advanced(
                    model, self.test_dl, threshold_info, self.device, is_baseline=mc["is_baseline"])

                trial_metrics['acc'].append(acc * 100); trial_metrics['recall'].append(recall * 100)
                trial_metrics['f1'].append(f1 * 100); trial_metrics['ece'].append(ece)
                trial_metrics['energy_red'].append(er * 100); trial_metrics['latency'].append(lat)

                if trial == self.cfg.num_trials - 1:
                    exit_df = collect_exit_statistics(model, self.test_dl, threshold_info, self.device, is_baseline=mc["is_baseline"])
                    last_analysis = analyze_exit_patterns(exit_df, model.num_stages)
                    last_exit_df = exit_df; last_ece_bins = ece_bins
                    print_analysis_report(last_analysis, name)

                print(f"  Trial {trial+1}: Acc={acc*100:.2f}%, F1={f1*100:.2f}%, ER={er*100:.2f}%")

            self.all_trial_metrics[name] = trial_metrics
            self.results.append({
                "Model": name, "Lambda": self.cfg.energy_lambda, "Params": total_params,
                "MFLOPs": f"{total_flops:.2f}",
                "Accuracy (%)": f"{np.mean(trial_metrics['acc']):.2f} +/- {np.std(trial_metrics['acc']):.2f}",
                "Recall (%)": f"{np.mean(trial_metrics['recall']):.2f} +/- {np.std(trial_metrics['recall']):.2f}",
                "F1 Score (%)": f"{np.mean(trial_metrics['f1']):.2f} +/- {np.std(trial_metrics['f1']):.2f}",
                "ECE": f"{np.mean(trial_metrics['ece']):.4f} +/- {np.std(trial_metrics['ece']):.4f}",
                "Energy Red (%)": f"{np.mean(trial_metrics['energy_red']):.2f} +/- {np.std(trial_metrics['energy_red']):.2f}",
                "Latency (ms)": f"{np.mean(trial_metrics['latency']):.2f} +/- {np.std(trial_metrics['latency']):.2f}",
            })

            exits = [0] * model.num_stages
            if last_exit_df is not None:
                for s in range(model.num_stages):
                    exits[s] = int((last_exit_df["exit_stage"] == s).sum())
            self.viz_data["exit_distributions"][name] = exits
            self.viz_data["per_model"][name] = {"ece_bin_data": last_ece_bins,
                "ece": np.mean(trial_metrics['ece']), "analysis": last_analysis, "exit_df": last_exit_df}

        if self.all_trial_metrics and self.cfg.num_trials >= 2:
            stats = compute_all_statistics(self.all_trial_metrics, list(self.all_trial_metrics.keys()))
            print_statistical_report(stats)
            self.viz_data["statistical_tests"] = stats

        df = pd.DataFrame(self.results)
        self.viz_data["results_df"] = df; self.viz_data["results_list"] = self.results
        return df

    def run_sizes(self):
        size_results = []
        for name, channels in SIZE_CONFIGS.items():
            mc = {"channels": channels, "is_baseline": False, "adaptive": False}
            model = self._build(mc)
            print(f"\n--- {name}: {model.count_parameters():,} params ---")
            set_seed(self.cfg.base_seed)
            ti = self._train(model, mc, self.cfg.base_seed)
            acc, _, f1, _, er, _, _, _ = evaluate_model_advanced(model, self.test_dl, ti, self.device, is_baseline=False)
            size_results.append({"name": name, "channels": channels, "params": model.count_parameters(),
                "mflops": sum(model.stage_flops) / 1e6, "accuracy": acc * 100, "f1": f1 * 100, "energy_reduction": er * 100})
            print(f"  Acc={acc*100:.2f}%, F1={f1*100:.2f}%, ER={er*100:.2f}%")
        self.viz_data["size_results"] = size_results
        return pd.DataFrame(size_results)

    def run_pruning(self, ratios=None):
        if ratios is None: ratios = [0.0, 0.1, 0.25, 0.5]
        pruning_results = []
        mc = {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False}
        set_seed(self.cfg.base_seed)
        base = self._build(mc)
        self._train(base, mc, self.cfg.base_seed)
        base_flops = sum(base.stage_flops)

        for ratio in ratios:
            print(f"\n--- Prune {ratio:.0%} ---")
            if ratio == 0.0:
                model, pf = base, base_flops
            else:
                model, _ = apply_structured_pruning(base, ratio, self.cfg.in_channels, self.cfg.seq_len, self.cfg.num_classes)
                model = model.to(self.device); pf = sum(model.stage_flops)
                cfn = self._criterion_fn(model, mc)
                opt = optim.Adam(model.parameters(), lr=self.cfg.learning_rate * 0.1)
                train_joint(model, self.train_dl, 5, opt, cfn, self.device, energy_lambda=self.cfg.energy_lambda)

            ti = calibrate_thresholds(model, self.val_dl, self.device, strategy=self.cfg.threshold_strategy)
            acc, _, f1, _, er, _, _, _ = evaluate_model_advanced(model, self.test_dl, ti, self.device, is_baseline=False)
            fr = (1.0 - pf / base_flops) * 100
            pruning_results.append({"prune_ratio": ratio, "params": model.count_parameters(),
                "accuracy": acc * 100, "f1": f1 * 100, "flops_reduction": fr, "energy_reduction": er * 100})
            print(f"  Acc={acc*100:.2f}%, FLOPs Red={fr:.1f}%")

        self.viz_data["pruning_results"] = pruning_results
        return pd.DataFrame(pruning_results)

    def run_strategies(self):
        strategy_results = {}
        mc = {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False}
        set_seed(self.cfg.base_seed)
        model = self._build(mc)
        self._train(model, mc, self.cfg.base_seed)

        for strat in ["confidence", "entropy", "patience"]:
            ti = calibrate_thresholds(model, self.val_dl, self.device, strategy=strat,
                target_acc=self.cfg.target_acc, entropy_percentile=self.cfg.entropy_percentile, patience=self.cfg.patience)
            acc, _, f1, ece, er, lat, _, _ = evaluate_model_advanced(model, self.test_dl, ti, self.device, is_baseline=False)
            strategy_results[strat] = {"accuracy": acc*100, "f1": f1*100, "energy_reduction": er*100, "ece": ece, "latency_ms": lat}
            print(f"  {strat}: Acc={acc*100:.2f}%, F1={f1*100:.2f}%, ER={er*100:.2f}%")

        self.viz_data["strategy_results"] = strategy_results
        return strategy_results

    def save(self):
        if self.results:
            pd.DataFrame(self.results).to_csv(os.path.join(self.cfg.results_dir, "main_results.csv"), index=False)

        out = {"config": vars(self.cfg), "results_summary": self.results,
               "raw_trial_metrics": {n: {k: [float(v) for v in vs] for k, vs in m.items()}
                                     for n, m in self.all_trial_metrics.items()}}
        if "statistical_tests" in self.viz_data:
            stats = self.viz_data["statistical_tests"]
            js = {}
            if "confidence_intervals" in stats: js["confidence_intervals"] = stats["confidence_intervals"]
            if "pairwise_comparisons" in stats:
                js["pairwise_comparisons"] = {m: [{k: v for k, v in c.items()} for c in cs]
                                               for m, cs in stats["pairwise_comparisons"].items()}
            out["statistical_analysis"] = js

        with open(os.path.join(self.cfg.results_dir, "experiment_results.json"), 'w') as f:
            json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Early Exit Research — Bonn EEG")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-sizes", action="store_true")
    parser.add_argument("--run-pruning", action="store_true")
    parser.add_argument("--run-strategies", action="store_true")
    parser.add_argument("--run-tuning", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--strategy", type=str, default="confidence", choices=["confidence", "entropy", "patience"])
    args = parser.parse_args()

    cfg = Config(base_seed=args.seed, num_trials=args.trials, threshold_strategy=args.strategy)
    pipe = Pipeline(cfg)
    inspect_data(pipe.train_dl)

    run_any = args.run_ablation or args.run_sizes or args.run_pruning or args.run_strategies or args.run_tuning or args.run_all
    if not run_any:
        args.run_ablation = True

    if args.run_all or args.run_ablation:
        print("\n" + "="*50 + "\nARCHITECTURE ABLATION\n" + "="*50)
        print(pipe.run_ablation().to_string())

    if args.run_all or args.run_strategies:
        print("\n" + "="*50 + "\nTHRESHOLD STRATEGIES\n" + "="*50)
        pipe.run_strategies()

    if args.run_all or args.run_sizes:
        print("\n" + "="*50 + "\nMODEL SIZE SCALING\n" + "="*50)
        print(pipe.run_sizes().to_string())

    if args.run_all or args.run_pruning:
        print("\n" + "="*50 + "\nSTRUCTURED PRUNING\n" + "="*50)
        print(pipe.run_pruning().to_string())

    if args.run_all or args.run_tuning:
        print("\n" + "="*50 + "\nHYPERPARAMETER SEARCH\n" + "="*50)
        from tuning import HyperparameterSearchSpace, run_hyperparameter_search
        mc = {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False}
        search = HyperparameterSearchSpace(learning_rates=[5e-4, 1e-3, 5e-3], energy_lambdas=[0.01, 0.05, 0.1],
            weight_decays=[1e-4, 1e-3], warmup_epochs_list=[8, 12], joint_epochs_list=[8, 12])
        pipe.viz_data["tuning_results"] = run_hyperparameter_search(
            model_builder_fn=lambda: pipe._build(mc),
            criterion_builder_fn=lambda model, el, sl: EnergyJointLoss(model.stage_flops, pipe.class_weights, energy_lambda=el),
            train_dl=pipe.train_dl, val_dl=pipe.val_dl, test_dl=pipe.test_dl,
            search_space=search, device=pipe.device, max_combinations=30, seed=args.seed, results_dir=cfg.results_dir)

    generate_all_plots(pipe.viz_data, cfg.plots_dir)
    pipe.save()
    print("\nDone.")
