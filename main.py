import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import from modules
from data_utils import get_dataloaders
from models import GenericEarlyExitNet, AdaptiveEarlyExitNet
from losses import EnergyJointLoss, AdaptiveEnergyJointLoss
from trainer import train_classifiers_only, train_joint, calibrate_thresholds
from evaluator import evaluate_model_advanced
from utils import inspect_data

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
        
        inspect_data(self.train_dl)

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

        df = pd.DataFrame(self.results)
        return df

if __name__ == "__main__":
    # Instantiate Configuration and execute
    cfg = ExperimentConfig(num_trials=2, warmup_epochs=2, joint_epochs=1) # Reduced epochs for testing if needed
    pipeline = ResearchPipeline(cfg)
    df_results = pipeline.run_ablation(energy_lambdas=[0.01, 0.05])
    
    # Save results locally instead of just printing or using colab display
    os.makedirs("./results", exist_ok=True)
    df_results.to_csv("./results/ablation_results.csv", index=False)
    
    print("\n--- Final Results ---")
    print(df_results.to_string())
    print("\nResults saved to ./results/ablation_results.csv")
