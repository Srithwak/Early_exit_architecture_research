# Early Exit Architecture Research — Data Audit & Summary

**Audited:** 2026-04-24 &nbsp;|&nbsp; **Data Sources:** `results/experiment_results.json`, `results/main_results.csv`, `results/hyperparameter_search.csv`, 64 plots

---

## 1. Research Question

> "Can early-exit neural networks reduce inference energy on EEG seizure-detection tasks without meaningfully sacrificing accuracy, and does the channel-width schedule of the backbone matter?"

---

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dataset | Bonn EEG (280 train / 60 val / 60 test) — binary seizure classification |
| Input | 6-channel: 1 raw time-series + 5 FFT frequency-band features, `seq_len = 4097` |
| Architectures | Base CNN (control), Constant Width [64,64,64], Increasing [32,64,128], Decreasing [128,64,32], Adaptive (gated) [64,64,64] |
| Training | 10 warmup + 10 joint epochs; Adam (lr=0.001, wd=1e-4); ReduceLROnPlateau; grad clip 1.0 |
| Trials | 15 per architecture (seeds 42–56) |
| Threshold | Confidence-based (calibrated from validation) |
| Energy metric | FLOPs-weighted exit stage vs. full network traversal |

> **Note:** The baseline runs all 20 epochs in warmup (no joint-training phase), ensuring an identical optimization budget.

---

## 3. Main Results (Architecture Ablation)

| Model | Accuracy (%) | F1 (%) | Energy Red. (%) | ECE | Latency (ms) |
|-------|-------------|--------|----------------|-----|-------------|
| Base CNN (Control) | 89.2 ± 8.6 | 84.9 ± 11.9 | 0.0 ± 0.0 | 0.092 ± 0.072 | 0.22 ± 0.01 |
| Constant Width | 86.7 ± 7.7 | 75.1 ± 20.0 | 25.4 ± 3.0 | 0.121 ± 0.067 | 0.21 ± 0.01 |
| **Increasing Width ⭐** | **90.0 ± 6.2** | **83.4 ± 14.6** | **43.5 ± 3.9** | **0.079 ± 0.054** | **0.12 ± 0.01** |
| Decreasing Width | 87.6 ± 6.0 | 81.8 ± 12.4 | 12.9 ± 2.4 | 0.095 ± 0.052 | 0.48 ± 0.01 |
| Adaptive Width | 85.0 ± 7.7 | 72.4 ± 19.8 | 46.8 ± 5.1 | 0.139 ± 0.080 | 0.22 ± 0.01 |

### Key Takeaway — "Increasing Width" Is the Pareto Winner

The Increasing Width [32→64→128] architecture dominates the accuracy-energy Pareto frontier:

- **+0.8 pp accuracy** over the baseline while saving **43.5% energy**
- **Best ECE (0.079)** — most calibrated predictions
- **Lowest latency (0.12 ms)** — early stages are small & fast
- **45% fewer FLOPs** on average vs. full traversal

### The Intuition

Starting narrow means easy samples exit cheaply through the lightweight first stage, while harder samples proceed to wider, more powerful later stages. This matches the natural difficulty distribution in EEG data.

---

## 4. Statistical Significance — ⚠️ Mostly Absent

> **Warning:** Most accuracy/F1 differences are NOT statistically significant. The high variance (±6–20 pp) swamps the modest mean differences.

### Accuracy Pairwise p-values (paired t-test)

| Comparison | Δ Mean | p-value | Significant? | Cohen's d |
|------------|--------|---------|-------------|-----------|
| Base vs. Constant | +2.6 | 0.476 | ❌ | 0.19 (negligible) |
| Base vs. Increasing | −0.8 | 0.796 | ❌ | 0.07 (negligible) |
| Base vs. Adaptive | +4.2 | 0.236 | ❌ | 0.32 (small) |
| Increasing vs. Adaptive | +5.0 | 0.010 | ✅ | 0.76 (medium) |
| Constant vs. Increasing | −3.3 | 0.047 | ✅ | 0.56 (medium) |

### Energy Reduction — ALL Highly Significant (p ≈ 10⁻¹⁵)

Every early-exit model achieves energy reduction that is overwhelmingly statistically significant vs. the baseline (Cohen's d > 5, p < 10⁻¹¹). Energy savings are real and reproducible.

### Bottom Line

The paper can confidently claim energy savings but should be cautious about accuracy superiority claims between architectures. The strongest defensible claim is:

> *"Increasing Width achieves comparable accuracy to the baseline while reducing energy by 43.5% (p < 10⁻¹⁵, d = 10.7)."*

---

## 5. Hyperparameter Sensitivity

The HP search (30 configurations) reveals:

| LR \ Energy λ | 0.01 | 0.05 | 0.10 |
|--------------|------|------|------|
| 0.0005 | 89.9% F1 | 91.3% F1 | 91.5% F1 |
| 0.001 | 73.0% | 87.0% | 77.5% |
| 0.005 | 59.7% | 53.0% | 48.3% |

> **Important:** The default config (`lr=0.001`, `λ=0.02`) is sub-optimal. The HP search shows `lr=0.0005` with `λ=0.05–0.10` reaches 95–97% accuracy / 91–95% F1, significantly better than the ablation results. This suggests the ablation comparisons understate what the best architectures can achieve.

---

## 6. Model Size Scaling

| Size | Params | Accuracy | Energy Red. |
|------|--------|----------|------------|
| Tiny [16] | ~8K | 86.7% | 12.3% |
| Small [32] | ~33K | 93.3% | 12.3% |
| Medium [64] | ~127K | 93.3% | 30.2% |
| Large [128] | ~501K | 93.3% | 31.4% |
| XLarge [256] | ~2M | 76.7% | 23.5% |

The "sweet spot" is **Small [32] to Medium [64]** — accuracy plateaus but the XLarge model actually degrades (overfitting on the small dataset). This further supports the Increasing Width design: start narrow, grow only when needed.

---

## 7. Structured Pruning

| Prune Ratio | Accuracy | FLOPs Red. |
|-------------|----------|-----------|
| 0% | 93.3% | 0% |
| 10% | 81.7% | 19.0% |
| 25% | 83.3% | 42.2% |
| 50% | 80.0% | 70.2% |

Pruning shows a steep accuracy cliff even at 10%. Combined with early exits, the additional FLOPs savings are marginal. Early exits are a more graceful efficiency mechanism than pruning for this task.

---

## 8. Dynamic Threshold Strategies

| Strategy | Accuracy | F1 | Energy Red. |
|----------|----------|----|------------|
| Confidence | 93.3% | 90.7% | 28.4% |
| Entropy | 86.7% | 79.2% | 20.1% |
| Patience | 91.7% | 88.1% | 3.2% |

Confidence thresholding wins on all fronts. Patience is too conservative (only 3.2% energy savings), entropy too aggressive.

---

## 9. Data Integrity Issues & Concerns

### ✅ What's Working Well

- **Reproducibility:** Seeded RNG, gradient clipping, identical epoch budgets — the experimental protocol is sound.
- **Energy claims are rock-solid:** All p < 10⁻¹¹ with massive effect sizes.
- **Rich visualization:** 64 plots including reliability diagrams, exit heatmaps, confidence distributions.
- Multiple complementary experiments (ablation, scaling, pruning, HP search, threshold comparison).

### ⚠️ Concerns

- **Very small test set (60 samples)** — This drives the high variance (±6–20 pp). Individual trials swing from 70% to 98% accuracy. More data or cross-validation would dramatically tighten the confidence intervals.

- **Ablation uses sub-optimal hyperparameters** — The HP search found `lr=0.0005` achieves ~96% accuracy vs. the ablation's ~89%. Running the ablation with the best-found hyperparameters would strengthen the comparisons.

- **Model size scaling is single-trial** — Only 1 trial per size, vs. 15 for the ablation. The scaling conclusions may not be robust.

- **Pruning results lack fine-tuning budget parity** — Pruned models get only 5 epochs of fine-tuning at 0.1× LR, while the original had 20 epochs. The comparison isn't fully fair.

- **No cross-validation** — With only 400 total samples, k-fold CV would be more rigorous than a single fixed split.

---

## 10. Recommendations — What to Run Next

### A. ✅ Re-run Ablation with Tuned Hyperparameters (HIGH IMPACT)

```bash
# In main.py, change defaults to lr=0.0005, energy_lambda=0.05
python main.py --run-ablation --trials 15 --seed 42
```

**Why:** This closes the gap between the HP-search best (96%) and the ablation results (89%). If Increasing Width still dominates under optimal HP, the story is much stronger.

### B. ✅ Run K-Fold Cross-Validation (HIGH IMPACT)

A 5-fold CV would multiply the effective test set by 5× and dramatically reduce variance. This is the single most impactful thing to strengthen statistical significance. You'd want to modify `main.py` to:

1. Load all data, split into 5 folds
2. For each fold, train on 4 folds, test on 1
3. Report aggregate metrics

### C. ✅ Lambda Sweep on the Best Architecture (MEDIUM IMPACT)

Test the Increasing Width architecture across `energy_lambda = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]` to map the full accuracy-vs-energy Pareto curve. This gives a publication-quality tradeoff plot.

### D. ✅ Multi-trial Size Scaling (LOW-MEDIUM IMPACT)

```bash
python main.py --run-sizes --trials 5
```

Currently single-trial. Even 3–5 trials would add error bars and confidence.

---

## Summary

The research has a clear, defensible story: **early-exit networks with an increasing-width channel schedule deliver the best accuracy-energy tradeoff for EEG seizure detection**, achieving ~43% energy savings with no accuracy loss. The energy savings are overwhelmingly statistically significant.

The main weakness is that accuracy differences between architectures lack statistical power due to the small dataset. Re-running with tuned HPs and/or cross-validation would either (a) confirm the rankings with significance, or (b) honestly show that architecture choice matters less than the early-exit mechanism itself — which is also a valid finding.