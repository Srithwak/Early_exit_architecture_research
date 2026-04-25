# 🎤 Presentation Guide: Early-Exit Neural Network Architectures
### Energy-Efficient Seizure Detection from EEG Signals
**Target Duration: 10 minutes | 12 slides**

---

## Slide-by-Slide Breakdown

---

### SLIDE 1 — Title Slide ⏱️ ~30s

**Slide Text:**
> # Early-Exit Neural Network Architectures for Energy-Efficient Seizure Detection
> **Rithwak Somepalli · Suryaprakash Murugavvel · Monique Gaye · Amrutha Kodali**
>
> CS [Course Number] — Spring 2026

**Visual:** Clean title layout with a subtle brain/EEG waveform graphic in the background. Keep it minimal and professional.

**What to say:**
> "Hi everyone, today we're presenting our research on early-exit neural network architectures for energy-efficient seizure detection from EEG signals. This project explores how we can make neural networks smarter about *when to stop computing* — particularly for a medical application where efficiency really matters."

---

### SLIDE 2 — Problem & Motivation ⏱️ ~1 min

**Slide Text:**
> ## The Problem
> - **1 in 26 people** will develop epilepsy in their lifetime
> - Wearable EEG devices need **real-time** seizure detection
> - Standard deep networks process every input through all layers — **wasteful**
> - Easy cases (clearly non-seizure) get the same compute as ambiguous ones
>
> ## Our Insight
> *"Not all inputs are equally hard — the network should decide when it's confident enough to stop."*

**Visual:** A diagram showing two paths:
1. Traditional CNN: input → [Layer 1] → [Layer 2] → [Layer 3] → prediction (always full depth)
2. Early-exit CNN: input → [Layer 1] → **EXIT** ✓ (for easy samples) vs input → [Layer 1] → [Layer 2] → [Layer 3] → prediction (for hard samples)

**What to say:**
> "So here's the problem. About 1 in 26 people will develop epilepsy. For wearable EEG devices to be practical for continuous monitoring, they need to be energy-efficient. But traditional deep learning models process every single input through the entire network, whether it's an obvious non-seizure recording or an ambiguous edge case. That's wasteful.
>
> Our core insight is simple: not all inputs are equally hard. If the network is already confident at an early layer, why make it keep computing? Early-exit architectures attach classifier heads at intermediate layers, so the model can stop as soon as it's confident enough."

---

### SLIDE 3 — Our Contributions ⏱️ ~45s

**Slide Text:**
> ## Contributions
> 1. **Two novel architectures**: Decreasing-Width and Adaptive-Width early-exit CNNs
> 2. **Comprehensive comparison** of 5 architectures under identical, bias-controlled conditions
> 3. **Three dynamic thresholding strategies**: Confidence, Entropy, and Patience
> 4. **Structured pruning** for additional compression beyond early exit
> 5. **Statistical rigor**: 15-trial experiments with paired t-tests, Wilcoxon tests, Cohen's d, and 95% CIs

**Visual:** A simple numbered list with icons next to each item (lightbulb for novel, scale for comparison, gauge for thresholding, scissors for pruning, chart for stats).

**What to say:**
> "Let me highlight what we actually did. First, we proposed two novel architectures — a decreasing-width design and an adaptive-width design that dynamically prunes features. Second, we conducted a comprehensive, fair comparison of five architectures under identical training conditions, with 15 trials per model and full statistical testing. We also compared three different strategies for deciding when to exit, and explored structured pruning as an additional compression technique."

---

### SLIDE 4 — The Five Architectures ⏱️ ~1 min 15s

**Slide Text:**
> ## Architecture Design Space
>
> | Architecture | Channel Progression | Novel? |
> |---|---|---|
> | **Baseline CNN** | [64, 64, 64] — no exits | Control |
> | **Constant Width** | [64, 64, 64] + exits | Existing |
> | **Increasing Width** | [32, 64, 128] + exits | Existing |
> | **Decreasing Width** | [128, 64, 32] + exits | **Ours** |
> | **Adaptive Width** | [64, 64, 64] + channel gates + exits | **Ours** |

**Visual:** An architecture diagram showing the five models side by side, with colored blocks representing layer widths. The decreasing model should look like a funnel (wide→narrow), increasing like an expanding shape, and adaptive with "gate" symbols between layers. Annotate the two novel ones clearly.

**What to say:**
> "Here are our five architectures. The baseline is a standard 3-stage CNN with no exits — it always uses all layers. The constant-width and increasing-width designs are established approaches in the literature. Our novel contributions are the decreasing-width model — think of it as a funnel that captures broad EEG patterns in early, wide layers, then narrows down to focus on the most discriminative features — and the adaptive-width model, which uses learned channel gates that dynamically suppress irrelevant features at each stage. The hypothesis is that early layers need to cast a wide net across EEG frequency bands, while deeper layers should focus."

---

### SLIDE 5 — Methodology Deep Dive ⏱️ ~1 min

**Slide Text:**
> ## Training Pipeline
> 1. **Phase 1 — Warmup** (10 epochs): Train classifiers only, exits frozen, λ=0
> 2. **Phase 2 — Joint** (10 epochs): Train everything with energy-aware loss (λ=0.02)
> 3. **Phase 3 — Calibrate**: Tune exit thresholds on validation set
>
> ## Energy-Aware Loss Function
> $$\mathcal{L} = \sum_{i} p_{\text{reach}}^{(i)} \cdot p_{\text{exit}}^{(i)} \cdot \mathcal{L}_{\text{CE}}^{(i)} + \lambda \cdot p_{\text{reach}}^{(i)} \cdot c_{\text{FLOP}}^{(i)}$$
>
> ## Fairness Controls
> - All models: same total epochs, same seeds, same data splits
> - Baseline gets equivalent epoch budget (all warmup)
> - Weighted cross-entropy for class imbalance

**Visual:** A flow diagram: `Data → Phase 1 (Warmup) → Phase 2 (Joint Training with Energy Loss) → Phase 3 (Threshold Calibration) → Evaluation`. Include the loss equation.

**What to say:**
> "Our training pipeline has three phases. First, a warmup phase where we only train the classifier heads — exit policies are frozen. This ensures every classifier branch learns good representations. Then, joint training with our energy-aware loss function. This loss balances classification accuracy against computational cost — the lambda term penalizes using deeper layers when early exits would suffice. Finally, we calibrate the exit thresholds on a held-out validation set.
>
> Crucially, to keep the comparison fair, all five models train under identical conditions — same seeds, same epoch budget, same data splits. The baseline gets the same total epochs but entirely in warmup mode."

---

### SLIDE 6 — Dataset & Preprocessing ⏱️ ~30s

**Slide Text:**
> ## Bonn EEG Dataset
> - **400 total samples** (280 train / 60 val / 60 test)
> - Binary classification: seizure vs. non-seizure
> - Sampling rate: 173.6 Hz, sequence length: 4,097 timepoints
>
> ## Preprocessing
> - Z-score normalization per sample
> - FFT → 5 frequency band powers (δ, θ, α, β, γ) tiled as extra channels
> - Input shape: **(6, 4097)** — 1 raw + 5 frequency bands

**Visual:** A small waveform plot of EEG data, plus a diagram showing the 6-channel input (raw signal + 5 frequency bands stacked).

**What to say:**
> "We use the Bonn EEG seizure dataset — it's small but clean, with 400 samples split into training, validation, and test sets. Each sample is a 4,097-point EEG recording. We preprocess by normalizing each sample, then computing FFT to extract power across five standard EEG frequency bands — delta through gamma. These are tiled as additional channels, giving us a 6-channel input to the convolutional network."

---

### SLIDE 7 — Main Results ⏱️ ~1 min 15s

**Slide Text:**
> ## Architecture Comparison (15 trials, mean ± std)
>
> | Model | Accuracy | F1 Score | Energy Reduction |
> |---|---|---|---|
> | Baseline CNN | 89.2 ± 8.6% | 84.9 ± 11.9% | 0% |
> | Constant Width | 86.7 ± 7.7% | 75.1 ± 20.0% | **25.4%** |
> | **Increasing Width** | **90.0 ± 6.2%** | **83.4 ± 14.6%** | **43.5%** |
> | Decreasing Width | 87.6 ± 6.0% | 81.8 ± 12.8% | 12.9% |
> | Adaptive Width | 85.0 ± 7.7% | 72.4 ± 19.8% | **46.8%** |

**Visual:** Use the **accuracy_vs_energy.png** Pareto plot — it's your single most powerful figure. Consider also showing the results table beside it.

**What to say:**
> "Here are our main results. The key takeaway is this Pareto plot on the right. The x-axis is energy reduction, the y-axis is accuracy. You want to be in the upper right — high accuracy AND high energy savings.
>
> The Increasing Width model is the clear Pareto-optimal choice — it achieves the highest accuracy at 90%, which actually *matches* the baseline, while saving 43.5% of the compute. That's remarkable — nearly half the energy with zero accuracy loss.
>
> The Adaptive Width model achieves the most energy savings at 46.8%, but at a cost — accuracy drops to 85%. Our novel Decreasing Width model maintains competitive accuracy at 87.6% but its funnel design means early exits handle less of the feature space, so energy savings are modest at 12.9%.
>
> Importantly, none of the accuracy differences between early-exit models and the baseline are statistically significant by paired t-test, meaning we're getting energy savings essentially for free."

---

### SLIDE 8 — Exit Behavior Analysis ⏱️ ~1 min

**Slide Text:**
> ## Where Do Samples Exit?
> - Most samples exit at **Stage 0** (the earliest point)
> - Increasing Width: 85%+ exit at Stage 0 with 83% optimal exit decisions
> - Only 5% "underthinking" (exited early but wrong) — acceptable
> - 11.7% "overthinking" (correct earlier but exited later) — room to improve thresholds
>
> ## Key Insight
> *EEG seizure detection is a "mostly easy" problem — the majority of inputs can be classified with minimal computation.*

**Visual:** Two images side-by-side:
1. The **exit_distributions_grid.png** (shows where samples exit for each model)
2. The **exit_quality_Increasing_Width.png** (overthinking/underthinking breakdown)

**What to say:**
> "This is one of the most interesting findings. When we look at where samples actually exit, the overwhelming majority leave at Stage 0 — the very first exit point. For the Increasing Width model, only about 5% of samples are 'underthinking,' meaning they exited early but got the answer wrong. About 12% are 'overthinking' — they were correct at an earlier stage but went deeper unnecessarily. And 83% are optimal — they exited at the right time.
>
> This tells us something important about the problem domain: EEG seizure detection, at least on this dataset, is a 'mostly easy' problem. Most recordings are clearly non-seizure, and the network figures that out almost immediately."

---

### SLIDE 9 — Dynamic Thresholding & Calibration ⏱️ ~45s

**Slide Text:**
> ## Three Exit Strategies Compared
> | Strategy | Accuracy | Energy Red. | F1 Score |
> |---|---|---|---|
> | **Confidence** | **93.3%** | **28.4%** | **90.7%** |
> | Entropy | 86.7% | 20.1% | 79.2% |
> | Patience | 91.7% | 3.2% | 88.1% |
>
> - **Confidence-based** is the best overall tradeoff
> - **Patience** is too conservative — barely saves energy
> - Models are well-calibrated (ECE = 0.079 for Increasing Width)

**Visual:** The **threshold_comparison.png** (3-panel side-by-side bars) plus the **reliability_Increasing_Width.png** calibration diagram.

**What to say:**
> "We compared three strategies for deciding when to exit. Confidence-based exiting — stop when the softmax probability exceeds a threshold — wins on all metrics. Entropy-based is more aggressive but less accurate. Patience-based, where you wait for consecutive stages to agree, is too conservative and barely saves energy.
>
> We also measured calibration using Expected Calibration Error. The Increasing Width model has an ECE of just 0.079, meaning when it says it's 90% confident, it really is right about 90% of the time. This is critical for medical applications."

---

### SLIDE 10 — Pruning & Scaling ⏱️ ~45s

**Slide Text:**
> ## Structured Pruning: Additional Compression
> - Remove least important channels (L1-norm ranking)
> - Creates genuinely smaller networks (not masked)
> - 10% pruning: accuracy 93.3% → 81.7% but **19% FLOPs reduction**
> - 25% pruning: accuracy → 83.3%, **43% FLOPs reduction**
> - 50% pruning: accuracy → 80%, **70% FLOPs reduction**
>
> ## Model Size Scaling
> - Sweet spot: **Small [32] to Medium [64]** — 93%+ accuracy, 25-30% energy savings
> - XLarge [256] overfits badly (77% accuracy with 1.9M params)

**Visual:** Two plots side-by-side:
1. **pruning_impact.png** (accuracy vs FLOPs reduction)
2. **model_size_scaling.png** (dual-axis accuracy + energy vs params)

**What to say:**
> "Beyond early exits, we explored structured pruning — physically removing the least important convolutional channels. At 25% pruning, you lose about 10 points of accuracy but save 43% of compute, on top of early-exit savings.
>
> Our model size scaling study shows a clear sweet spot around 32-64 channels. Bigger isn't always better — the XLarge model with 1.9 million parameters actually overfits and drops to 77% accuracy. For this dataset, lean and efficient wins."

---

### SLIDE 11 — Conclusions ⏱️ ~45s

**Slide Text:**
> ## Key Findings
> 1. **Early exits can save 43-47% energy** with negligible accuracy loss
> 2. **Increasing Width is Pareto-optimal**: highest accuracy (90%) + 43.5% energy savings
> 3. Adaptive Width achieves maximum energy savings (47%) but sacrifices accuracy
> 4. **Confidence-based thresholding** is the best exit strategy
> 5. Most EEG samples are "easy" — classifiable at the first exit stage
> 6. Structured pruning provides **complementary** compression on top of early exits
>
> ## Broader Impact
> *These techniques could enable longer battery life for wearable seizure monitors, making continuous EEG monitoring more practical for patients with epilepsy.*

**Visual:** A summary graphic — maybe a stylized bar chart highlighting the key numbers (90% accuracy, 43.5% energy savings, 15 trials) with the Pareto plot as a small thumbnail.

**What to say:**
> "So to wrap up our findings: Early-exit architectures can save nearly half the computational energy with negligible accuracy loss. The Increasing Width architecture is the Pareto-optimal design for this task. Our novel architectures revealed interesting tradeoffs — the decreasing-width model's funnel design limits early-exit opportunities, while the adaptive model aggressively saves energy but at an accuracy cost. Confidence-based thresholding is the winner, and the models are well-calibrated. Perhaps most importantly, we showed that the vast majority of EEG samples are 'easy' and can be classified almost immediately."

---

### SLIDE 12 — Team Contributions & Future Work ⏱️ ~30s

**Slide Text:**
> ## Team Contributions
> | Name | Contributions |
> |---|---|
> | **Rithwak Somepalli** | Novel architecture design & implementation (decreasing-width, adaptive-width), feature pruning, full pipeline development, statistical testing |
> | **Suryaprakash Murugavvel** | Baseline CNN + existing architectures, hyperparameter tuning, model evaluation |
> | **Monique Gaye** | Energy/memory analysis, exit threshold strategies, visualizations |
> | **Amrutha Kodali** | Data preprocessing, normalization, class balancing, cross-dataset evaluation |
>
> ## Future Work
> - Cross-dataset evaluation (Bonn → CHB-MIT)
> - Hardware deployment benchmarking on edge devices
> - Multi-class seizure classification

**Visual:** A simple team table plus a "Future Work" bullet list. Optional: add team photos if you have them.

**What to say:**
> "Here are each team member's contributions. [Read through briefly]. For future work, we'd like to validate on the much larger CHB-MIT dataset with 23 patients and 983 hours of data, benchmark on actual edge hardware, and extend to multi-class seizure classification. Thank you — happy to take questions."

---

## 🎯 Presentation Tips

1. **Rehearse to 9:30** — leave 30s buffer for natural pauses
2. **The Pareto plot (Slide 7) is your hero figure** — spend the most time here
3. **Don't read the tables** — point to the highlights and tell the story
4. **Anticipate questions:**
   - *"Why not test on CHB-MIT?"* → Dataset is 983 hours; outside project scope but on roadmap
   - *"Are the accuracy differences significant?"* → No! That's the point. Energy savings are free.
   - *"How does this compare to transformers?"* → Our focus is lightweight CNNs for edge devices; transformer-based exits are future work
   - *"Why does decreasing width save less energy?"* → Funnel design front-loads compute in wide early layers, so even Stage 0 exits are expensive
   - *"Would you use this in a real hospital?"* → The confidence calibration results are promising (ECE < 0.08), but clinical validation would need much larger datasets

---

---

# 🔍 Gap Analysis: Current State vs. Initial Plans

> **Note**: This section is NOT for the presentation — it's for your awareness.

## What's Missing (besides other dataset testing)

### 1. ❌ Cross-Dataset Evaluation (CHB-MIT)
The README and proposal explicitly planned "train on Bonn → test on CHB-MIT." This is the single biggest gap. The CHB-MIT dataset (23 patients, 983 hours, multi-channel) would validate generalization. Without it, all results are on a single small dataset.

### 2. ❌ README Results Table is Empty
The `README.md` results table (lines 34-40) still has placeholder dashes "—" instead of actual numbers. The CSV has the data — this just needs copying over. **Easy fix.**

### 3. ⚠️ Comparison Against Published Benchmarks
The roadmap item "Comparison against published benchmarks" is unchecked. You don't have any literature comparison showing how your accuracy/energy tradeoff compares to other early-exit papers (e.g., BranchyNet, SDN, or similar on EEG tasks).

### 4. ⚠️ Hyperparameter Tuning is Narrow
The HP search (hp_sensitivity.png) only covers a 3×3 grid of learning rate × energy lambda. The full grid with weight decay, epoch counts, and architecture-specific parameters (e.g., sparsity_lambda for adaptive) hasn't been explored systematically.

### 5. ⚠️ High Variance Across Trials
Standard deviations are large (e.g., Baseline accuracy 89.2 ± 8.6%, F1 scores with ±12-20% std). This is partly due to the small dataset (only 60 test samples), but it means confidence intervals are wide and most pairwise differences are not statistically significant. More data or cross-validation would strengthen claims.

### 6. ⚠️ Adaptive Width Underperformance
The novel Adaptive Width model underperforms significantly — it has the worst accuracy (85%) and F1 (72.4%) while only marginally beating Increasing Width on energy savings. The channel gating mechanism may need more tuning (sparsity_lambda, gate architecture) or a different training schedule to converge properly.

### 7. ⚠️ No Hardware/Latency Benchmarking on Real Edge Devices
Latency is measured in Python wall-clock time on CPU, not on actual target hardware (e.g., ARM Cortex-M, Raspberry Pi). The roadmap doesn't explicitly plan this, but for a "wearable EEG device" motivation, it would strengthen the narrative.

### 8. ✅ BUT — What's Actually Working Well
- All 5 architectures fully implemented and running
- 10 publication-quality plot types (64 files)
- Full statistical testing pipeline (t-tests, Wilcoxon, CIs, effect sizes)
- 3 threshold strategies implemented and compared
- Structured pruning with genuine weight transfer (not masking)
- Modular, well-organized codebase
- 15-trial experiments with seed control
- Reproducibility controls (CUDA determinism, gradient clipping, LR scheduling)

### Summary Priority List
| Priority | Gap | Effort | Impact |
|---|---|---|---|
| 🔴 High | Update README results table | 5 min | Low but looks bad if empty |
| 🟡 Medium | Literature comparison numbers | 1-2 hours | Strengthens paper framing |
| 🟡 Medium | Tune Adaptive Width model | 2-4 hours | Fixes underperforming novel contribution |
| 🔵 Low | Expand HP search grid | 4+ hours | Marginal improvement |
| 🔵 Low | Cross-validation instead of single split | 3+ hours | Reduces variance |
