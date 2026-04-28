import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0',
          '#00BCD4', '#FF5722', '#607D8B', '#795548', '#CDDC39']


def _save(fig, d, name):
    fig.savefig(os.path.join(d, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(d, f"{name}.pdf"), bbox_inches='tight')
    plt.close(fig)


def _sn(name):
    return name.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')


def plot_accuracy_vs_energy(df, d):
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


def plot_reliability_diagram(bins, name, ece, d):
    fig, ax = plt.subplots(figsize=(8, 8))
    centers = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins]
    w = 1.0 / len(bins) * 0.8
    ax.bar(centers, [b["accuracy"] for b in bins], width=w, alpha=0.7, color='#2196F3', edgecolor='white', label='Outputs')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect')
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability: {name}\nECE = {ece:.4f}", fontsize=14, fontweight='bold')
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    _save(fig, d, f"reliability_{_sn(name)}")


def plot_exit_heatmap(analysis, name, d):
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


def plot_per_stage_accuracy(analysis, name, d):
    accs = analysis["per_stage_accuracy"]; counts = analysis["per_stage_counts"]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar([f"S{i}" for i in range(len(accs))], [a*100 for a in accs], color=COLORS[:len(accs)])
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
    for bar, a, c in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{a*100:.1f}%\nn={c}", ha='center', fontsize=10)
    ax.set_title(f"Per-Stage Accuracy: {name}", fontsize=14, fontweight='bold')
    _save(fig, d, f"stage_accuracy_{_sn(name)}")


def plot_confidence_distributions(exit_df, name, d):
    ns = exit_df["exit_stage"].nunique()
    fig, axes = plt.subplots(1, max(ns, 1), figsize=(5*ns, 5), squeeze=False); axes = axes.flatten()
    for s in range(ns):
        ax = axes[s]; sd = exit_df[exit_df["exit_stage"] == s]
        c = sd[sd["is_correct"] == 1]["confidence"]; ic = sd[sd["is_correct"] == 0]["confidence"]
        if len(c) > 0: ax.hist(c, bins=20, alpha=0.6, color='#4CAF50', label='Correct', density=True)
        if len(ic) > 0: ax.hist(ic, bins=20, alpha=0.6, color='#E91E63', label='Incorrect', density=True)
        ax.set_title(f"Stage {s} (n={len(sd)})"); ax.legend()
    fig.suptitle(f"Confidence: {name}", fontsize=14, fontweight='bold'); plt.tight_layout()
    _save(fig, d, f"confidence_dist_{_sn(name)}")


def plot_overthinking_underthinking(analysis, name, d):
    s = analysis["summary"]
    ot, ut, total = s["overthinking_count"], s["underthinking_count"], s["total_samples"]
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [ot, ut, total - ot - ut]
    bars = ax.bar(['Overthink', 'Underthink', 'Optimal'], vals, color=['#FF9800', '#E91E63', '#4CAF50'])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{v} ({v/total*100:.1f}%)", ha='center', fontweight='bold')
    ax.set_ylabel("Samples"); ax.set_title(f"Exit Quality: {name}", fontsize=14, fontweight='bold')
    _save(fig, d, f"exit_quality_{_sn(name)}")


def plot_hp_sensitivity(results, d):
    if not results: return
    import pandas as pd
    df = pd.DataFrame(results)
    if "lr" not in df.columns or "energy_lambda" not in df.columns: return
    pivot = df.pivot_table(index="lr", columns="energy_lambda", values="f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"{v:.3f}" for v in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([f"{v:.4f}" for v in pivot.index])
    ax.set_xlabel("Energy Lambda"); ax.set_ylabel("Learning Rate")
    plt.colorbar(im, ax=ax, label="F1 (%)"); ax.set_title("HP Sensitivity", fontsize=14, fontweight='bold')
    _save(fig, d, "hp_sensitivity")


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

    if "results_df" in data: plot_accuracy_vs_energy(data["results_df"], plots_dir)
    if "exit_distributions" in data:
        plot_exit_distributions_grid(data["exit_distributions"], data.get("results_list", []), plots_dir)
    for name, md in data.get("per_model", {}).items():
        if "ece_bin_data" in md: plot_reliability_diagram(md["ece_bin_data"], name, md.get("ece", 0), plots_dir)
        if "analysis" in md:
            plot_exit_heatmap(md["analysis"], name, plots_dir)
            plot_per_stage_accuracy(md["analysis"], name, plots_dir)
            plot_overthinking_underthinking(md["analysis"], name, plots_dir)
        if "exit_df" in md: plot_confidence_distributions(md["exit_df"], name, plots_dir)
    if "tuning_results" in data: plot_hp_sensitivity(data["tuning_results"], plots_dir)
    if "size_results" in data: plot_model_size_scaling(data["size_results"], plots_dir)
    if "pruning_results" in data: plot_pruning_impact(data["pruning_results"], plots_dir)
    if "strategy_results" in data: plot_threshold_comparison(data["strategy_results"], plots_dir)
    print("Done.")
