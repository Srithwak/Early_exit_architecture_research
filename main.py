import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from dataclasses import dataclass

from dataset import get_dataloaders
from models import GenericEarlyExitNet, AdaptiveEarlyExitNet, EnergyJointLoss, AdaptiveEnergyJointLoss
from train import train_classifiers_only, train_joint, calibrate_thresholds
from evaluate import inspect_data, evaluate_model, evaluate_model_advanced

def run_pipeline():
    base_dir = "."
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(base_dir, "data", "bonn")
    train_dl, val_dl, test_dl, class_weights = get_dataloaders(data_dir, batch_size=64)
    class_weights = class_weights.to(device)

    # Display preprocessed data attributes
    inspect_data(train_dl)

    models_to_test = {
        "1. Base CNN (Control)": {"channels": [64, 64, 64], "is_baseline": True, "adaptive": False},
        "2. Constant Width": {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False},
        "3. Increasing Width": {"channels": [32, 64, 128], "is_baseline": False, "adaptive": False},
        "4. Decreasing Width": {"channels": [128, 64, 32], "is_baseline": False, "adaptive": False},
        "5. Adaptive Width": {"channels": [64, 64, 64], "is_baseline": False, "adaptive": True}
    }

    results = []
    exit_dists = {}
    class_exit_dists = {}

    for name, config in models_to_test.items():
        print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
        if config.get("adaptive"):
            model = AdaptiveEarlyExitNet(in_channels=6, channel_sizes=config["channels"], seq_len=4097).to(device)
        else:
            model = GenericEarlyExitNet(in_channels=6, channel_sizes=config["channels"], seq_len=4097).to(device)

        total_mflops = sum(model.stage_flops) / 1e6
        print(f"Total FLOPs: {total_mflops:.2f}M | Stage Breakdown: {[f'{f/1e6:.2f}M' for f in model.stage_flops]}")

        def criterion_fn(energy_lambda):
            if config.get("adaptive"):
                return AdaptiveEnergyJointLoss(model.stage_flops, class_weights, energy_lambda=energy_lambda, sparsity_lambda=0.01, is_baseline=config["is_baseline"])
            return EnergyJointLoss(model.stage_flops, class_weights, energy_lambda=energy_lambda, is_baseline=config["is_baseline"])

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        if config["is_baseline"]:
            train_classifiers_only(model, train_dl, epochs=40, optimizer=optimizer, criterion_fn=criterion_fn, device=device)
            thresholds = [1.0] * (model.num_stages - 1)
        else:
            train_classifiers_only(model, train_dl, epochs=25, optimizer=optimizer, criterion_fn=criterion_fn, device=device)
            train_joint(model, train_dl, epochs=15, optimizer=optimizer, criterion_fn=criterion_fn, device=device, energy_lambda=0.02)
            thresholds = calibrate_thresholds(model, val_dl, device=device, target_acc=0.99)

        acc, recall, f1, energy_red, stage_exits, class_exits = evaluate_model(model, test_dl, thresholds, device, is_baseline=config["is_baseline"])

        results.append({
            "Model": name,
            "Accuracy (%)": acc * 100,
            "Recall (%)": recall * 100,
            "F1 Score (%)": f1 * 100,
            "Energy Reduction (%)": energy_red * 100
        })
        exit_dists[name] = stage_exits
        class_exit_dists[name] = class_exits

        print(f"Result -> Acc: {acc*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1*100:.2f}%, Energy Reduction: {energy_red*100:.2f}%")
        print(f"  Total Exits: {stage_exits}")

    df = pd.DataFrame(results)
    print("\nResults DataFrame:")
    print(df.to_string())

    # Plotting Exit Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    stages = [f"Stage {i+1}" for i in range(3)]

    for i, (name, exits) in enumerate(exit_dists.items()):
        axes[i].bar(stages, exits, color=['skyblue', 'orange', 'green'])
        axes[i].set_title(f"{name}\nAcc: {results[i]['Accuracy (%)']:.1f}% | Energy Red: {results[i]['Energy Reduction (%)']:.1f}%")
        axes[i].set_ylabel("Samples")

    if len(exit_dists) < len(axes):
        for j in range(len(exit_dists), len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_comparisons_exits.png"))
    # plt.show() # Commented out to avoid blocking execution

    # Plotting Accuracy vs Energy Reduction Trade-off
    plt.figure(figsize=(8,6))
    for idx, row in df.iterrows():
        plt.scatter(row["Energy Reduction (%)"], row["Accuracy (%)"], s=150, label=row["Model"])
        plt.text(row["Energy Reduction (%)"]+0.5, row["Accuracy (%)"], row["Model"].split(".")[1], fontsize=9)

    plt.title("Accuracy vs Energy Reduction Trade-off")
    plt.xlabel("Theoretical MFLOPs Reduction (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "energy_vs_acc.png"))
    # plt.show() # Commented out to avoid blocking execution


@dataclass
class ExperimentConfig:
    """Centralized configuration for the research experiments."""
    data_dir: str = "./data/bonn"
    batch_size: int = 64
    seq_len: int = 4097
    use_freq_bands: bool = True  # Enable our new FFT features
    in_channels: int = 6         # 1 Raw Time-Series + 5 Frequency Bands
    num_trials: int = 2
    warmup_epochs: int = 8
    joint_epochs: int = 4
    target_acc: float = 0.98
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

class ResearchPipeline:
    """Modular pipeline for managing and aggregating Early-Exit experiments."""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dl, self.val_dl, self.test_dl, self.class_weights = get_dataloaders(
            self.config.data_dir, self.config.batch_size, use_freq_bands=self.config.use_freq_bands
        )
        self.class_weights = self.class_weights.to(self.device)
        self.results = []

        # Dictionary defining the architectures to test
        self.models_to_test = {
            "1. Constant Width": {"channels": [64, 64, 64], "is_baseline": False, "adaptive": False},
            "2. Adaptive Width": {"channels": [64, 64, 64], "is_baseline": False, "adaptive": True}
        }

    def _build_model(self, config_dict):
        """Factory method to construct the appropriate network."""
        if config_dict.get("adaptive"):
            return AdaptiveEarlyExitNet(in_channels=self.config.in_channels, channel_sizes=config_dict["channels"], seq_len=self.config.seq_len).to(self.device)
        return GenericEarlyExitNet(in_channels=self.config.in_channels, channel_sizes=config_dict["channels"], seq_len=self.config.seq_len).to(self.device)

    def _get_criterion(self, model, config_dict, e_lambda):
        """Factory method to construct the objective function."""
        if config_dict.get("adaptive"):
            return AdaptiveEnergyJointLoss(model.stage_flops, self.class_weights, energy_lambda=e_lambda, sparsity_lambda=0.01, is_baseline=config_dict["is_baseline"])
        return EnergyJointLoss(model.stage_flops, self.class_weights, energy_lambda=e_lambda, is_baseline=config_dict["is_baseline"])

    def run_ablation(self, energy_lambdas):
        """Runs the entire matrix of experiments across trials and parameters."""
        for e_lambda in energy_lambdas:
            print(f"\n{'='*60}\nRunning Ablation: Energy Lambda = {e_lambda}\n{'='*60}")

            for name, m_config in self.models_to_test.items():
                print(f"\nEvaluating {name} over {self.config.num_trials} trials...")
                trial_metrics = {k: [] for k in ['acc', 'recall', 'f1', 'ece', 'energy_red', 'latency']}

                for trial in range(self.config.num_trials):
                    print(f"  -> Trial {trial+1}/{self.config.num_trials}")
                    model = self._build_model(m_config)
                    optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

                    # Closure for loss function
                    def criterion_fn(energy_lambda):
                        return self._get_criterion(model, m_config, energy_lambda)

                    # Pipeline Execution
                    train_classifiers_only(model, self.train_dl, self.config.warmup_epochs, optimizer, criterion_fn, self.device)
                    train_joint(model, self.train_dl, self.config.joint_epochs, optimizer, criterion_fn, self.device, energy_lambda=e_lambda)
                    thresholds = calibrate_thresholds(model, self.val_dl, self.device, self.config.target_acc)

                    acc, recall, f1, ece, energy_red, latency = evaluate_model_advanced(model, self.test_dl, thresholds, self.device, is_baseline=m_config["is_baseline"])

                    # Record metrics
                    trial_metrics['acc'].append(acc * 100)
                    trial_metrics['recall'].append(recall * 100)
                    trial_metrics['f1'].append(f1 * 100)
                    trial_metrics['ece'].append(ece)
                    trial_metrics['energy_red'].append(energy_red * 100)
                    trial_metrics['latency'].append(latency)

                # Aggregate and format
                self.results.append({
                    "Model": name,
                    "Lambda": e_lambda,
                    "Accuracy (%)": f"{np.mean(trial_metrics['acc']):.2f} ± {np.std(trial_metrics['acc']):.2f}",
                    "Recall (%)": f"{np.mean(trial_metrics['recall']):.2f} ± {np.std(trial_metrics['recall']):.2f}",
                    "F1 Score (%)": f"{np.mean(trial_metrics['f1']):.2f} ± {np.std(trial_metrics['f1']):.2f}",
                    "ECE": f"{np.mean(trial_metrics['ece']):.4f} ± {np.std(trial_metrics['ece']):.4f}",
                    "Energy Red (%)": f"{np.mean(trial_metrics['energy_red']):.2f} ± {np.std(trial_metrics['energy_red']):.2f}",
                    "Latency (ms)": f"{np.mean(trial_metrics['latency']):.2f} ± {np.std(trial_metrics['latency']):.2f}"
                })

        return pd.DataFrame(self.results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Early Exit Experiments")
    parser.add_argument("--run-pipeline", action="store_true", help="Run the primary model comparisons pipeline")
    parser.add_argument("--run-ablation", action="store_true", help="Run the advanced ablation research pipeline")
    
    args = parser.parse_args()
    
    if args.run_pipeline or not args.run_ablation:
        print("Running Primary Pipeline...")
        run_pipeline()
        
    if args.run_ablation:
        print("\nRunning Ablation Research Pipeline...")
        cfg = ExperimentConfig(num_trials=2, warmup_epochs=8, joint_epochs=4)
        pipeline = ResearchPipeline(cfg)
        df_results = pipeline.run_ablation(energy_lambdas=[0.01, 0.05])
        print("\nResults DataFrame:")
        print(df_results.to_string())
