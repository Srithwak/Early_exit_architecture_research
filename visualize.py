import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0',
          '#00BCD4', '#FF5722', '#607D8B', '#795548', '#CDDC39']


def _save(fig, d, name):
    fig.savefig(os.path.join(d, f"{name}.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)


def _sn(name):
    return name.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')


def plot_accuracy_vs_energy(df, d):
    """Core finding: accuracy vs energy tradeoff across architectures."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, row in df.iterrows():
        a = str(row.get("Accuracy (%)", "0")).replace("+/-", "±")
        e = str(row.get("Energy Red (%)", "0")).replace("+/-", "±")
        try:
            am, astd = float(a.split("±")[0]), (float(a.split("±")[1]) if "±" in a else 0)
            em, estd = float(e.split("±")[0]), (float(e.split("±")[1]) if "±" in e else 0)
        except: am = astd = em = estd = 0
        ax.errorbar(em, am, xerr=estd, yerr=astd, fmt='o', markersize=10, capsize=5,
                     color=COLORS[idx % len(COLORS)], label=row.get("Model", f"Model {idx}"))
    ax.set_xlabel("Energy Reduction (%)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Energy Reduction", fontsize=15, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    _save(fig, d, "accuracy_vs_energy")


def plot_combined_reliability(per_model_data, d):
    """Single combined reliability diagram for all models."""
    models_with_bins = {n: md for n, md in per_model_data.items()
                        if md.get("ece_bin_data")}
    if not models_with_bins:
        return

    n = len(models_with_bins)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, (name, md) in enumerate(models_with_bins.items()):
        ax = axes[i]
        bins = md["ece_bin_data"]
        ece = md.get("ece", 0)
        centers = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins]
        w = 1.0 / len(bins) * 0.8
        ax.bar(centers, [b["accuracy"] for b in bins], width=w, alpha=0.7,
               color=COLORS[i % len(COLORS)], edgecolor='white')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_title(f"{name}\nECE={ece:.4f}", fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Calibration Reliability Diagrams", fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, d, "combined_reliability")


def plot_combined_exit_quality(per_model_data, d):
    """Single combined overthink/underthink chart for all models."""
    models_with_analysis = {n: md for n, md in per_model_data.items()
                            if md.get("analysis") and md["analysis"].get("summary")}
    if not models_with_analysis:
        return

    names = list(models_with_analysis.keys())
    ot_rates = [models_with_analysis[n]["analysis"]["summary"]["overthinking_rate"] * 100 for n in names]
    ut_rates = [models_with_analysis[n]["analysis"]["summary"]["underthinking_rate"] * 100 for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, ot_rates, w, label='Overthinking', color='#FF9800', alpha=0.85)
    bars2 = ax.bar(x + w/2, ut_rates, w, label='Underthinking', color='#E91E63', alpha=0.85)

    for bar, v in zip(bars1, ot_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')
    for bar, v in zip(bars2, ut_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel("Rate (%)"); ax.legend()
    ax.set_title("Exit Quality: Overthinking vs Underthinking", fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, d, "combined_exit_quality")


def plot_exit_heatmap(analysis, name, d):
    """Class-level exit distribution heatmap for a single model."""
    pc = analysis["per_class_exits"]; classes = sorted(pc.keys())
    ns = len(list(pc.values())[0])
    data = np.array([pc[c] for c in classes], dtype=float)
    rs = data.sum(axis=1, keepdims=True); rs[rs == 0] = 1; dn = data / rs * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(dn, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(ns)); ax.set_xticklabels([f"S{i}" for i in range(ns)])
    ax.set_yticks(range(len(classes))); ax.set_yticklabels([f"C{c}" for c in classes])
    for i in range(len(classes)):
        for j in range(ns):
            ax.text(j, i, f"{dn[i,j]:.1f}%", ha='center', va='center', fontsize=10)
    ax.set_title(f"Exit Distribution: {name}", fontsize=14, fontweight='bold')
    _save(fig, d, f"exit_heatmap_{_sn(name)}")


def plot_model_size_scaling(results, d):
    if not results: return
    fig, ax1 = plt.subplots(figsize=(10, 6))
    p = [r["params"] for r in results]; a = [r["accuracy"] for r in results]; e = [r["energy_reduction"] for r in results]
    ax1.plot(p, a, 'o-', color='#2196F3', ms=10, lw=2, label='Accuracy')
    ax1.set_xlabel("Parameters"); ax1.set_ylabel("Accuracy (%)", color='#2196F3')
    ax2 = ax1.twinx()
    ax2.plot(p, e, 's-', color='#FF9800', ms=10, lw=2, label='Energy Red')
    ax2.set_ylabel("Energy Red (%)", color='#FF9800')
    ax1.set_title("Model Size Scaling", fontsize=14, fontweight='bold')
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc='best')
    _save(fig, d, "model_size_scaling")


def plot_pruning_impact(results, d):
    if not results: return
    fig, ax1 = plt.subplots(figsize=(10, 6))
    r = [x["prune_ratio"] for x in results]; a = [x["accuracy"] for x in results]; f = [x["flops_reduction"] for x in results]
    ax1.plot(r, a, 'o-', color='#4CAF50', ms=10, lw=2, label='Accuracy')
    ax1.set_xlabel("Prune Ratio"); ax1.set_ylabel("Accuracy (%)", color='#4CAF50')
    ax2 = ax1.twinx()
    ax2.plot(r, f, 's-', color='#E91E63', ms=10, lw=2, label='FLOPs Red')
    ax2.set_ylabel("FLOPs Red (%)", color='#E91E63')
    ax1.set_title("Structured Pruning Impact", fontsize=14, fontweight='bold')
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc='best')
    _save(fig, d, "pruning_impact")


def plot_threshold_comparison(results, d):
    if not results: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    strats = list(results.keys())
    for ax, m, t in zip(axes, ["accuracy", "energy_reduction", "f1"], ["Accuracy (%)", "Energy Red (%)", "F1 (%)"]):
        vals = [results[s].get(m, 0) for s in strats]
        bars = ax.bar(strats, vals, color=COLORS[:len(strats)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{v:.1f}", ha='center', fontweight='bold')
        ax.set_ylabel(t); ax.set_title(t, fontweight='bold')
    fig.suptitle("Threshold Strategy Comparison", fontsize=15, fontweight='bold'); plt.tight_layout()
    _save(fig, d, "threshold_comparison")


def plot_exit_distributions_grid(dists, results, d):
    n = len(dists); cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n == 1: axes = np.array([axes])
    axes = axes.flatten()
    for i, (name, exits) in enumerate(dists.items()):
        axes[i].bar([f"S{j}" for j in range(len(exits))], exits, color=COLORS[:len(exits)])
        axes[i].set_title(name, fontweight='bold'); axes[i].set_ylabel("Samples")
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout()
    _save(fig, d, "exit_distributions_grid")


def generate_all_plots(data, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\nGenerating plots -> {plots_dir}/")

    # Aggregate comparison plots
    if "results_df" in data:
        plot_accuracy_vs_energy(data["results_df"], plots_dir)
    if "exit_distributions" in data:
        plot_exit_distributions_grid(data["exit_distributions"], data.get("results_list", []), plots_dir)

    # Combined per-model plots (one chart each instead of 5 separate ones)
    per_model = data.get("per_model", {})
    if per_model:
        plot_combined_reliability(per_model, plots_dir)
        plot_combined_exit_quality(per_model, plots_dir)

    # Exit heatmaps for the two most informative models only
    heatmap_targets = ["Constant Width", "Decreasing Width"]
    for name, md in per_model.items():
        if name in heatmap_targets and md.get("analysis"):
            plot_exit_heatmap(md["analysis"], name, plots_dir)

    # Experiment-specific plots
    if "size_results" in data:
        plot_model_size_scaling(data["size_results"], plots_dir)
    if "pruning_results" in data:
        plot_pruning_impact(data["pruning_results"], plots_dir)
    if "strategy_results" in data:
        plot_threshold_comparison(data["strategy_results"], plots_dir)

    print("Done.")
