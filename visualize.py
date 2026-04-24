"""
visualize.py — Publication-quality visualizations for early-exit research.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0',
          '#00BCD4', '#FF5722', '#607D8B', '#795548', '#CDDC39']


def _save_plot(fig, plots_dir, name):
    fig.savefig(os.path.join(plots_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(plots_dir, f"{name}.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png/.pdf")


def plot_accuracy_vs_energy(results_df, plots_dir):
    """1. Pareto frontier of accuracy vs energy reduction with error bars."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, row in results_df.iterrows():
        acc_str = str(row.get("Accuracy (%)", "0"))
        er_str = str(row.get("Energy Red (%)", "0"))
        try:
            acc_mean = float(acc_str.split("±")[0].strip())
            acc_std = float(acc_str.split("±")[1].strip()) if "±" in acc_str else 0
            er_mean = float(er_str.split("±")[0].strip())
            er_std = float(er_str.split("±")[1].strip()) if "±" in er_str else 0
        except (ValueError, IndexError):
            acc_mean, acc_std, er_mean, er_std = 0, 0, 0, 0
        ax.errorbar(er_mean, acc_mean, xerr=er_std, yerr=acc_std,
                     fmt='o', markersize=10, capsize=5, color=COLORS[idx % len(COLORS)],
                     label=row.get("Model", f"Model {idx}"))
    ax.set_xlabel("Energy Reduction (%)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs Energy Reduction Trade-off", fontsize=15, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    _save_plot(fig, plots_dir, "accuracy_vs_energy")


def plot_reliability_diagram(ece_bin_data, model_name, ece_value, plots_dir):
    """2. Calibration / Reliability diagram."""
    fig, ax = plt.subplots(figsize=(8, 8))
    bins_acc = [b["accuracy"] for b in ece_bin_data]
    bins_conf = [b["confidence"] for b in ece_bin_data]
    bins_count = [b["count"] for b in ece_bin_data]
    bin_centers = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in ece_bin_data]
    width = 1.0 / len(ece_bin_data) * 0.8

    ax.bar(bin_centers, bins_acc, width=width, alpha=0.7, color='#2196F3',
           edgecolor='white', label='Outputs')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    ax.set_xlabel("Mean Predicted Confidence", fontsize=13)
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=13)
    ax.set_title(f"Reliability Diagram: {model_name}\nECE = {ece_value:.4f}",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    safe_name = model_name.replace(" ", "_").replace(".", "")
    _save_plot(fig, plots_dir, f"reliability_{safe_name}")


def plot_exit_heatmap(analysis_results, model_name, plots_dir):
    """3. Exit distribution heatmap (classes × stages)."""
    per_class = analysis_results["per_class_exits"]
    classes = sorted(per_class.keys())
    num_stages = len(list(per_class.values())[0])

    data = np.array([per_class[c] for c in classes], dtype=float)
    # Normalize per class
    row_sums = data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    data_norm = data / row_sums * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data_norm, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(num_stages))
    ax.set_xticklabels([f"Stage {i}" for i in range(num_stages)])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"Class {c}" for c in classes])
    for i in range(len(classes)):
        for j in range(num_stages):
            ax.text(j, i, f"{data_norm[i,j]:.1f}%\n({int(data[i,j])})",
                    ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax, label="Exit %")
    ax.set_title(f"Exit Distribution: {model_name}", fontsize=14, fontweight='bold')
    safe_name = model_name.replace(" ", "_").replace(".", "")
    _save_plot(fig, plots_dir, f"exit_heatmap_{safe_name}")


def plot_per_stage_accuracy(analysis_results, model_name, plots_dir):
    """4. Per-stage accuracy breakdown."""
    accs = analysis_results["per_stage_accuracy"]
    counts = analysis_results["per_stage_counts"]
    stages = [f"Stage {i}" for i in range(len(accs))]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    bars = ax1.bar(stages, [a * 100 for a in accs], color=COLORS[:len(accs)],
                   edgecolor='white', linewidth=1.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=13)
    ax1.set_ylim(0, 105)
    for bar, acc, count in zip(bars, accs, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{acc*100:.1f}%\n(n={count})", ha='center', fontsize=10)
    ax1.set_title(f"Per-Stage Accuracy: {model_name}", fontsize=14, fontweight='bold')
    safe_name = model_name.replace(" ", "_").replace(".", "")
    _save_plot(fig, plots_dir, f"stage_accuracy_{safe_name}")


def plot_confidence_distributions(exit_df, model_name, plots_dir):
    """5. Confidence distributions at each exit (correct vs incorrect)."""
    num_stages = exit_df["exit_stage"].nunique()
    fig, axes = plt.subplots(1, max(num_stages, 1), figsize=(5 * num_stages, 5), squeeze=False)
    axes = axes.flatten()

    for s in range(num_stages):
        ax = axes[s]
        stage_df = exit_df[exit_df["exit_stage"] == s]
        correct = stage_df[stage_df["is_correct"] == 1]["confidence"]
        incorrect = stage_df[stage_df["is_correct"] == 0]["confidence"]
        if len(correct) > 0:
            ax.hist(correct, bins=20, alpha=0.6, color='#4CAF50', label='Correct', density=True)
        if len(incorrect) > 0:
            ax.hist(incorrect, bins=20, alpha=0.6, color='#E91E63', label='Incorrect', density=True)
        ax.set_title(f"Stage {s} (n={len(stage_df)})", fontsize=12)
        ax.set_xlabel("Confidence")
        ax.legend()

    fig.suptitle(f"Confidence Distributions: {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace(".", "")
    _save_plot(fig, plots_dir, f"confidence_dist_{safe_name}")


def plot_overthinking_underthinking(analysis_results, model_name, plots_dir):
    """6. Overthinking/underthinking summary."""
    summary = analysis_results["summary"]
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['Overthinking\n(correct earlier,\nexited later)',
                   'Underthinking\n(exited early,\nincorrect)',
                   'Optimal\n(correct at\nexit stage)']
    ot = summary["overthinking_count"]
    ut = summary["underthinking_count"]
    total = summary["total_samples"]
    optimal = total - ot - ut
    values = [ot, ut, optimal]
    colors_bar = ['#FF9800', '#E91E63', '#4CAF50']

    bars = ax.bar(categories, values, color=colors_bar, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val} ({val/total*100:.1f}%)", ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel("Number of Samples", fontsize=13)
    ax.set_title(f"Exit Quality: {model_name}", fontsize=14, fontweight='bold')
    safe_name = model_name.replace(" ", "_").replace(".", "")
    _save_plot(fig, plots_dir, f"exit_quality_{safe_name}")


def plot_hp_sensitivity(tuning_results, plots_dir):
    """7. Hyperparameter sensitivity heatmap (lr vs energy_lambda -> F1)."""
    if not tuning_results:
        return
    import pandas as pd
    df = pd.DataFrame(tuning_results)
    if "lr" not in df.columns or "energy_lambda" not in df.columns:
        return
    pivot = df.pivot_table(index="lr", columns="energy_lambda", values="f1", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.3f}" for v in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.4f}" for v in pivot.index])
    ax.set_xlabel("Energy Lambda", fontsize=13)
    ax.set_ylabel("Learning Rate", fontsize=13)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=9,
                        color='white' if val < pivot.values[~np.isnan(pivot.values)].mean() else 'black')
    plt.colorbar(im, ax=ax, label="F1 Score (%)")
    ax.set_title("Hyperparameter Sensitivity: LR vs Energy Lambda", fontsize=14, fontweight='bold')
    _save_plot(fig, plots_dir, "hp_sensitivity")


def plot_model_size_scaling(size_results, plots_dir):
    """8. Accuracy and energy vs model parameter count."""
    if not size_results:
        return
    fig, ax1 = plt.subplots(figsize=(10, 6))
    params = [r["params"] for r in size_results]
    accs = [r["accuracy"] for r in size_results]
    energy = [r["energy_reduction"] for r in size_results]
    names = [r["name"] for r in size_results]

    ax1.plot(params, accs, 'o-', color='#2196F3', markersize=10, linewidth=2, label='Accuracy (%)')
    ax1.set_xlabel("Parameters", fontsize=13)
    ax1.set_ylabel("Accuracy (%)", fontsize=13, color='#2196F3')
    ax2 = ax1.twinx()
    ax2.plot(params, energy, 's-', color='#FF9800', markersize=10, linewidth=2, label='Energy Red (%)')
    ax2.set_ylabel("Energy Reduction (%)", fontsize=13, color='#FF9800')
    for p, a, e, n in zip(params, accs, energy, names):
        ax1.annotate(n, (p, a), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    ax1.set_title("Model Size Scaling", fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    _save_plot(fig, plots_dir, "model_size_scaling")


def plot_pruning_impact(pruning_results, plots_dir):
    """9. Accuracy vs prune ratio."""
    if not pruning_results:
        return
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ratios = [r["prune_ratio"] for r in pruning_results]
    accs = [r["accuracy"] for r in pruning_results]
    flops_red = [r["flops_reduction"] for r in pruning_results]

    ax1.plot(ratios, accs, 'o-', color='#4CAF50', markersize=10, linewidth=2, label='Accuracy (%)')
    ax1.set_xlabel("Prune Ratio", fontsize=13)
    ax1.set_ylabel("Accuracy (%)", fontsize=13, color='#4CAF50')
    ax2 = ax1.twinx()
    ax2.plot(ratios, flops_red, 's-', color='#E91E63', markersize=10, linewidth=2, label='FLOPs Red (%)')
    ax2.set_ylabel("FLOPs Reduction (%)", fontsize=13, color='#E91E63')
    ax1.set_title("Impact of Structured Pruning", fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    _save_plot(fig, plots_dir, "pruning_impact")


def plot_threshold_comparison(strategy_results, plots_dir):
    """10. Side-by-side comparison of threshold strategies."""
    if not strategy_results:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    strategies = list(strategy_results.keys())
    metrics = ["accuracy", "energy_reduction", "f1"]
    titles = ["Accuracy (%)", "Energy Reduction (%)", "F1 Score (%)"]

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [strategy_results[s].get(metric, 0) for s in strategies]
        bars = ax.bar(strategies, vals, color=COLORS[:len(strategies)], edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')

    fig.suptitle("Dynamic Threshold Strategy Comparison", fontsize=15, fontweight='bold')
    plt.tight_layout()
    _save_plot(fig, plots_dir, "threshold_comparison")


def plot_exit_distributions_grid(all_exit_dists, all_results, plots_dir):
    """Grid of exit distribution bar charts for all models."""
    n = len(all_exit_dists)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, exits) in enumerate(all_exit_dists.items()):
        stages = [f"Stage {j}" for j in range(len(exits))]
        axes[i].bar(stages, exits, color=COLORS[:len(exits)], edgecolor='white')
        axes[i].set_title(name, fontsize=11, fontweight='bold')
        axes[i].set_ylabel("Samples")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    _save_plot(fig, plots_dir, "exit_distributions_grid")


def generate_all_plots(results_data, plots_dir):
    """Master function to generate all available plots from collected data."""
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\n{'='*50}\nGenerating Visualizations\n{'='*50}")

    if "results_df" in results_data:
        plot_accuracy_vs_energy(results_data["results_df"], plots_dir)

    if "exit_distributions" in results_data:
        plot_exit_distributions_grid(results_data["exit_distributions"],
                                     results_data.get("results_list", []), plots_dir)

    for model_name, data in results_data.get("per_model", {}).items():
        if "ece_bin_data" in data:
            plot_reliability_diagram(data["ece_bin_data"], model_name,
                                     data.get("ece", 0), plots_dir)
        if "analysis" in data:
            plot_exit_heatmap(data["analysis"], model_name, plots_dir)
            plot_per_stage_accuracy(data["analysis"], model_name, plots_dir)
            plot_overthinking_underthinking(data["analysis"], model_name, plots_dir)
        if "exit_df" in data:
            plot_confidence_distributions(data["exit_df"], model_name, plots_dir)

    if "tuning_results" in results_data:
        plot_hp_sensitivity(results_data["tuning_results"], plots_dir)
    if "size_results" in results_data:
        plot_model_size_scaling(results_data["size_results"], plots_dir)
    if "pruning_results" in results_data:
        plot_pruning_impact(results_data["pruning_results"], plots_dir)
    if "strategy_results" in results_data:
        plot_threshold_comparison(results_data["strategy_results"], plots_dir)

    print(f"All plots saved to {plots_dir}/")
