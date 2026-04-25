"""
run_tuned_ablation.py — Re-run architecture ablation with HP-search-optimized parameters.

The original ablation used lr=0.001, energy_lambda=0.02, which the HP search
showed is sub-optimal. The best config found was lr=0.0005, energy_lambda=0.05.

Results are saved to results/tuned_* to preserve originals.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ExperimentConfig, ResearchPipeline
from evaluate import inspect_data


def main():
    cfg = ExperimentConfig(
        # Tuned hyperparameters from HP search
        learning_rate=0.0005,
        energy_lambda=0.05,
        weight_decay=0.0001,
        # Keep same structure
        warmup_epochs=12,
        joint_epochs=8,
        # Statistical rigor
        num_trials=15,
        base_seed=42,
        threshold_strategy="confidence",
    )

    # Save to separate files so originals are preserved
    cfg.results_dir = "./results_tuned"

    print("=" * 60)
    print("TUNED ABLATION STUDY")
    print(f"  lr={cfg.learning_rate}, lambda={cfg.energy_lambda}")
    print(f"  warmup={cfg.warmup_epochs}, joint={cfg.joint_epochs}")
    print(f"  trials={cfg.num_trials}")
    print("=" * 60)

    pipeline = ResearchPipeline(cfg)
    inspect_data(pipeline.train_dl)

    df = pipeline.run_architecture_ablation()
    print("\n" + df.to_string())

    pipeline.generate_visualizations()
    pipeline.save_results()
    print("\n[DONE] Tuned ablation complete. Results in results_tuned/")


if __name__ == "__main__":
    main()
