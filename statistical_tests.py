"""
statistical_tests.py — Statistical significance testing for research rigor.

Provides paired comparisons between models, effect size calculations,
confidence intervals, and formatted reporting suitable for academic papers.
"""

import numpy as np
from scipy import stats
from itertools import combinations


def paired_ttest(scores_a, scores_b, alpha=0.05):
    """
    Perform a paired t-test between two sets of scores.
    
    Args:
        scores_a: list/array of metric values for model A (one per trial)
        scores_b: list/array of metric values for model B (one per trial)
        alpha: significance level
    
    Returns:
        dict with t-statistic, p-value, significant (bool), effect_size (Cohen's d)
    """
    a, b = np.array(scores_a), np.array(scores_b)
    
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False,
                "effect_size": 0.0, "note": "Too few trials for statistical test"}
    
    t_stat, p_value = stats.ttest_rel(a, b)
    
    # Cohen's d for paired samples
    diff = a - b
    d = diff.mean() / (diff.std(ddof=1) + 1e-10)
    
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "effect_size": float(d),
        "effect_magnitude": _interpret_cohens_d(abs(d)),
    }


def wilcoxon_test(scores_a, scores_b, alpha=0.05):
    """
    Non-parametric Wilcoxon signed-rank test.
    More appropriate for small sample sizes where normality can't be assumed.
    """
    a, b = np.array(scores_a), np.array(scores_b)
    diff = a - b
    
    if len(a) < 2 or np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "note": "Insufficient data or no difference"}
    
    try:
        stat, p_value = stats.wilcoxon(a, b)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
        }
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "note": "Wilcoxon test failed (identical distributions?)"}


def confidence_interval(scores, confidence=0.95):
    """
    Compute confidence interval for the mean.
    Uses t-distribution for small samples.
    """
    arr = np.array(scores)
    n = len(arr)
    if n < 2:
        return {"mean": float(arr.mean()), "ci_lower": float(arr.mean()),
                "ci_upper": float(arr.mean()), "margin": 0.0}
    
    mean = arr.mean()
    se = stats.sem(arr)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se
    
    return {
        "mean": float(mean),
        "ci_lower": float(mean - margin),
        "ci_upper": float(mean + margin),
        "margin": float(margin),
        "std": float(arr.std(ddof=1)),
        "n": n,
    }


def _interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def run_pairwise_comparisons(trial_results, metric="acc", alpha=0.05):
    """
    Run all pairwise comparisons between models.
    
    Args:
        trial_results: dict of model_name -> list of per-trial metric values
        metric: which metric to compare
        alpha: significance level
    
    Returns:
        list of comparison dicts
    """
    model_names = list(trial_results.keys())
    comparisons = []
    
    for name_a, name_b in combinations(model_names, 2):
        scores_a = trial_results[name_a]
        scores_b = trial_results[name_b]
        
        t_result = paired_ttest(scores_a, scores_b, alpha)
        w_result = wilcoxon_test(scores_a, scores_b, alpha)
        
        comparisons.append({
            "model_a": name_a,
            "model_b": name_b,
            "metric": metric,
            "mean_a": float(np.mean(scores_a)),
            "mean_b": float(np.mean(scores_b)),
            "diff": float(np.mean(scores_a) - np.mean(scores_b)),
            "paired_ttest": t_result,
            "wilcoxon": w_result,
        })
    
    return comparisons


def format_significance_table(comparisons):
    """Format pairwise comparisons as a readable table string."""
    lines = []
    lines.append(f"{'Model A':<22} {'Model B':<22} {'Diff':>8} {'p-value':>10} {'Sig?':>6} {'Effect':>10}")
    lines.append("-" * 80)
    
    for c in comparisons:
        sig = "YES" if c["paired_ttest"]["significant"] else "no"
        effect = c["paired_ttest"].get("effect_magnitude", "n/a")
        lines.append(
            f"{c['model_a']:<22} {c['model_b']:<22} "
            f"{c['diff']:>+8.2f} {c['paired_ttest']['p_value']:>10.4f} "
            f"{sig:>6} {effect:>10}"
        )
    
    return "\n".join(lines)


def compute_all_statistics(all_trial_metrics, model_names):
    """
    Comprehensive statistical analysis across all models and metrics.
    
    Args:
        all_trial_metrics: dict of model_name -> dict of metric_name -> list of trial values
        model_names: ordered list of model names
    
    Returns:
        dict with confidence_intervals, pairwise_comparisons per metric
    """
    results = {
        "confidence_intervals": {},
        "pairwise_comparisons": {},
    }
    
    metrics = ["acc", "recall", "f1", "ece", "energy_red"]
    metric_labels = {
        "acc": "Accuracy (%)",
        "recall": "Recall (%)",
        "f1": "F1 Score (%)",
        "ece": "ECE",
        "energy_red": "Energy Reduction (%)",
    }
    
    for metric in metrics:
        # Confidence intervals
        ci_data = {}
        trial_data = {}
        for name in model_names:
            if name in all_trial_metrics and metric in all_trial_metrics[name]:
                scores = all_trial_metrics[name][metric]
                ci_data[name] = confidence_interval(scores)
                trial_data[name] = scores
        
        results["confidence_intervals"][metric] = ci_data
        
        # Pairwise comparisons (skip baseline for some metrics)
        if len(trial_data) >= 2:
            comparisons = run_pairwise_comparisons(trial_data, metric)
            results["pairwise_comparisons"][metric] = comparisons
    
    return results


def print_statistical_report(stats_results):
    """Print a formatted statistical analysis report."""
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    
    metric_labels = {
        "acc": "Accuracy (%)",
        "recall": "Recall (%)",
        "f1": "F1 Score (%)",
        "ece": "ECE",
        "energy_red": "Energy Reduction (%)",
    }
    
    # Confidence intervals
    print("\n--- 95% Confidence Intervals ---")
    for metric, ci_data in stats_results["confidence_intervals"].items():
        label = metric_labels.get(metric, metric)
        print(f"\n  {label}:")
        for name, ci in ci_data.items():
            print(f"    {name:<22}: {ci['mean']:.2f} [{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}] (n={ci['n']})")
    
    # Pairwise comparisons
    print("\n--- Pairwise Significance Tests ---")
    for metric, comparisons in stats_results["pairwise_comparisons"].items():
        label = metric_labels.get(metric, metric)
        print(f"\n  {label}:")
        print(f"  {format_significance_table(comparisons)}")
    
    print("\n" + "=" * 80)
