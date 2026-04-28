import numpy as np
from scipy import stats
from itertools import combinations


def paired_ttest(scores_a, scores_b, alpha=0.05):
    a, b = np.array(scores_a), np.array(scores_b)
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False, "effect_size": 0.0}

    t_stat, p_value = stats.ttest_rel(a, b)
    diff = a - b
    d = diff.mean() / (diff.std(ddof=1) + 1e-10)

    mag = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
    return {"t_stat": float(t_stat), "p_value": float(p_value),
            "significant": p_value < alpha, "effect_size": float(d), "effect_magnitude": mag}


def wilcoxon_test(scores_a, scores_b, alpha=0.05):
    a, b = np.array(scores_a), np.array(scores_b)
    if len(a) < 2 or np.all(a - b == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    try:
        stat, p = stats.wilcoxon(a, b)
        return {"statistic": float(stat), "p_value": float(p), "significant": p < alpha}
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}


def confidence_interval(scores, confidence=0.95):
    arr = np.array(scores)
    if len(arr) < 2:
        return {"mean": float(arr.mean()), "ci_lower": float(arr.mean()),
                "ci_upper": float(arr.mean()), "margin": 0.0, "std": 0.0, "n": len(arr)}

    mean = arr.mean()
    se = stats.sem(arr)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=len(arr) - 1)
    margin = t_crit * se
    return {"mean": float(mean), "ci_lower": float(mean - margin), "ci_upper": float(mean + margin),
            "margin": float(margin), "std": float(arr.std(ddof=1)), "n": len(arr)}


def run_pairwise_comparisons(trial_results, metric="acc", alpha=0.05):
    comparisons = []
    for name_a, name_b in combinations(trial_results.keys(), 2):
        sa, sb = trial_results[name_a], trial_results[name_b]
        t_result = paired_ttest(sa, sb, alpha)
        w_result = wilcoxon_test(sa, sb, alpha)
        comparisons.append({
            "model_a": name_a, "model_b": name_b, "metric": metric,
            "mean_a": float(np.mean(sa)), "mean_b": float(np.mean(sb)),
            "diff": float(np.mean(sa) - np.mean(sb)),
            "paired_ttest": t_result, "wilcoxon": w_result,
        })
    return comparisons


def compute_all_statistics(all_trial_metrics, model_names):
    results = {"confidence_intervals": {}, "pairwise_comparisons": {}}
    metrics = ["acc", "recall", "f1", "ece", "energy_red"]

    for metric in metrics:
        ci_data, trial_data = {}, {}
        for name in model_names:
            if name in all_trial_metrics and metric in all_trial_metrics[name]:
                scores = all_trial_metrics[name][metric]
                ci_data[name] = confidence_interval(scores)
                trial_data[name] = scores
        results["confidence_intervals"][metric] = ci_data
        if len(trial_data) >= 2:
            results["pairwise_comparisons"][metric] = run_pairwise_comparisons(trial_data, metric)

    return results


def print_statistical_report(stats_results):
    print("\n--- Statistical Tests ---")
    for metric, ci_data in stats_results["confidence_intervals"].items():
        print(f"\n  {metric}:")
        for name, ci in ci_data.items():
            print(f"    {name:<22}: {ci['mean']:.2f} [{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]")

    for metric, comps in stats_results["pairwise_comparisons"].items():
        print(f"\n  Pairwise ({metric}):")
        for c in comps:
            sig = "SIG" if c["paired_ttest"]["significant"] else "ns"
            print(f"    {c['model_a']:<20} vs {c['model_b']:<20} diff={c['diff']:+.2f} p={c['paired_ttest']['p_value']:.4f} {sig}")
