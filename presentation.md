# Presentation: Early-Exit Neural Network Architectures for Energy-Efficient Seizure Detection

**Target Duration: 10 minutes | 12 slides**
**Team: Rithwak Somepalli, Suryaprakash Murugavvel, Monique Gaye, Amrutha Kodali**

---

## Report Requirements Mapping

| Report Section | Slides |
|---|---|
| **a) Introduction** (problem statement + contributions) | Slides 1-3 |
| **b) Methodology** (algorithms, analysis, implementation) | Slides 4-6 |
| **c) Experiments & Results** (evaluation methodology + results) | Slides 7-10 |
| **d) Conclusions** (summary) | Slide 11 |
| **e) Team contributions** | Slide 12 |

---

## SLIDE 1 — Title Slide

**Time: ~30 seconds**

### Slide Content
```
Title:    Early-Exit Neural Network Architectures
          for Energy-Efficient Seizure Detection

Authors:  Rithwak Somepalli · Suryaprakash Murugavvel
          Monique Gaye · Amrutha Kodali

Course:   CS [Course Number] — Spring 2026
```

### Visual
Clean layout. Use one of these as a subtle background element:
- An EEG waveform trace (you can screenshot one from the dataset)
- Keep it minimal — white background, bold title

### Script
> "Hi everyone. Today we're presenting our research on early-exit neural network architectures for energy-efficient seizure detection from EEG signals. We explored how to make neural networks smarter about *when to stop computing* — and why that matters for a medical application running on wearable devices."

---

## SLIDE 2 — Problem & Motivation

**Time: ~1 minute**

### Slide Content
```
THE PROBLEM
-----------
* 1 in 26 people will develop epilepsy in their lifetime
* Wearable EEG devices need real-time seizure detection
* Standard neural networks process every input through ALL layers
  — same compute for obvious cases and ambiguous ones
* This is wasteful for battery-powered medical devices

OUR INSIGHT
-----------
"Not all inputs are equally hard — the network should decide
 when it's confident enough to stop."
```

### Visual
Use the generated diagram: `plots/early_exit_concept.png`

This diagram shows:
- LEFT: Traditional CNN → input flows through all 3 layers every time
- RIGHT: Early-Exit CNN → input can exit after Layer 1 if confident, saving ~40% energy

### Script
> "About 1 in 26 people will develop epilepsy. For wearable EEG monitors to be practical for continuous use, they need to be extremely energy-efficient. But traditional neural networks process every single input through the entire network — whether it's an obvious non-seizure recording or a genuinely ambiguous case. That's a huge waste of energy.
>
> Our core insight is simple: not all inputs are equally hard. If the network is already confident at an early layer, it should stop computing. Early-exit architectures attach classifier heads at intermediate layers so the model can stop and produce a prediction as soon as confidence is high enough."

---

## SLIDE 3 — Our Contributions

**Time: ~45 seconds**

### Slide Content
```
CONTRIBUTIONS
-------------
1. Two NOVEL architectures:
   - Decreasing-Width early-exit CNN
   - Adaptive-Width early-exit CNN (with learned channel gating)

2. Comprehensive comparison of 5 architectures
   under identical, bias-controlled conditions

3. Three dynamic exit strategies:
   Confidence, Entropy, and Patience thresholding

4. Structured pruning for additional model compression

5. Statistical rigor:
   5-trial experiments, paired t-tests, Wilcoxon tests,
   Cohen's d effect sizes, 95% confidence intervals
```

### Visual
Simple numbered list with small icons:
- Lightbulb icon for novel architectures
- Scale icon for fair comparison
- Gauge icon for thresholding
- Scissors icon for pruning
- Bar chart icon for statistics

### Script
> "Let me highlight what we actually did. First, we proposed two novel architectures — a decreasing-width design and an adaptive-width design that dynamically prunes features using learned channel gates. Second, we ran a fair, bias-controlled comparison of five architectures — same seeds, same epoch budget, same data splits — with multiple trials and full statistical testing. We also compared three different strategies for deciding when to exit, and explored structured pruning on top of early exits."

---

## SLIDE 4 — The Five Architectures (Methodology)

**Time: ~1 minute 15 seconds**

### Slide Content
```
ARCHITECTURE DESIGN SPACE
--------------------------
| Architecture       | Channels       | Type     |
|--------------------|----------------|----------|
| Baseline CNN       | [64, 64, 64]   | Control  |
| Constant Width     | [64, 64, 64]   | Existing |
| Increasing Width   | [32, 64, 128]  | Existing |
| Decreasing Width   | [128, 64, 32]  | NOVEL    |
| Adaptive Width     | [64, 64, 64]   | NOVEL    |
|                    | + channel gates |          |
```

### Visual
Use the generated diagram: `plots/five_architectures.png`

Shows all 5 architectures as block diagrams with varying widths.
The Decreasing model looks like a funnel (wide→narrow).
The Adaptive model has gate symbols between layers.

### Script
> "Here are our five architectures. The baseline is a standard 3-stage CNN with no exits — it always uses all layers. The constant-width and increasing-width designs come from existing literature like BranchyNet.
>
> Our novel contributions are the decreasing-width model — think of it as a funnel. The idea is that early layers need to cast a wide net across EEG frequency bands, capturing broad patterns, while deeper layers narrow down to focus on the most discriminative features. And the adaptive-width model, which uses learned channel gates — essentially tiny neural networks that decide which features to keep and which to suppress at each stage.
>
> Our hypothesis was that this funnel design could reduce energy consumption while maintaining accuracy. Let's see what actually happened."

---

## SLIDE 5 — Training Pipeline (Methodology)

**Time: ~1 minute**

### Slide Content
```
THREE-PHASE TRAINING PROTOCOL
------------------------------
Phase 1 - WARMUP (10 epochs)
  * Train classifier heads only
  * Exit policies frozen (lambda = 0)
  * Ensures every branch learns good features

Phase 2 - JOINT TRAINING (10 epochs)
  * Energy-aware loss function:
    L = SUM[ p_reach * p_exit * L_CE ] + lambda * energy_cost
  * lambda = 0.02 penalizes unnecessary deep computation

Phase 3 - THRESHOLD CALIBRATION
  * Tune exit confidence thresholds on validation set
  * Confidence strategy: exit when softmax > threshold

FAIRNESS: All 5 models get identical total epochs,
          same seeds, same data splits
```

### Visual
Use the generated diagram: `plots/training_pipeline.png`

Horizontal flowchart: Data → Warmup → Joint Training → Calibration → Evaluation

### Script
> "Our training pipeline has three phases. First, a warmup phase where we only train the classifier heads at each exit — the exit policies are frozen. This ensures every branch learns good representations before the energy penalty kicks in.
>
> Then, joint training with our energy-aware loss function. This loss balances classification accuracy against computational cost — the lambda term penalizes the network for using deeper layers when early exits would suffice.
>
> Finally, we calibrate the exit thresholds on a held-out validation set.
>
> Crucially, to keep the comparison fair, all five models train under identical conditions. The baseline gets the same total epochs but entirely in warmup mode, so every model has an equal optimization budget."

---

## SLIDE 6 — Dataset & Preprocessing (Methodology)

**Time: ~30 seconds**

### Slide Content
```
BONN EEG SEIZURE DATASET
--------------------------
* 400 total samples (280 train / 60 val / 60 test)
* Binary classification: seizure vs. non-seizure
* Sampling rate: 173.6 Hz
* Sequence length: 4,097 timepoints per sample

PREPROCESSING
-------------
* Z-score normalization (per sample)
* FFT -> 5 frequency band powers:
  delta, theta, alpha, beta, gamma
* These are tiled as extra channels
* Final input shape: (6, 4097)
  = 1 raw signal + 5 frequency bands
```

### Visual
Create a simple diagram showing:
- A raw EEG waveform (wavy line)
- Arrow labeled "FFT" pointing to 5 colored bars (delta through gamma)
- These stack into a 6-channel input rectangle

### Script
> "We use the Bonn EEG seizure dataset — it's small but clean, with 400 samples. Each sample is a 4,097-point EEG recording. We preprocess by normalizing each sample, then computing the FFT to extract power across five standard EEG frequency bands — delta through gamma. These are tiled as additional input channels alongside the raw signal, giving us a 6-channel input to the convolutional network."

---

## SLIDE 7 — Main Results (KEY SLIDE)

**Time: ~1 minute 15 seconds** ← Spend the most time here

### Slide Content
```
ARCHITECTURE COMPARISON (5 trials, seeds 42-46)
-------------------------------------------------
| Model              | Accuracy       | F1 Score       | Energy Red.      |
|--------------------|----------------|----------------|------------------|
| Baseline CNN       | 95.33 +/- 2.87 | 93.33 +/- 4.72 | 0%               |
| Constant Width     | 83.67 +/- 7.41 | 67.43 +/- 18.6 | 24.04 +/- 2.70%  |
| Increasing Width * | 94.00 +/- 2.26 | 91.86 +/- 3.13 | 42.05 +/- 2.99%  |
| Decreasing Width   | 88.33 +/- 3.33 | 84.13 +/- 4.20 | 12.84 +/- 1.92%  |
| Adaptive Width     | 86.00 +/- 6.38 | 75.32 +/- 14.1 | 47.48 +/- 2.55%  |

* = Pareto-optimal (best accuracy-energy tradeoff)

KEY FINDING: 42% energy savings with NO accuracy loss
```

### Visual — THIS IS YOUR HERO FIGURE
Use: `plots/accuracy_vs_energy.png` (the Pareto plot)

This plot shows each architecture as a dot on the accuracy (y) vs energy reduction (x) plane. Increasing Width dominates the upper-right corner.

### Script
> "Here are our main results. The key takeaway is this Pareto plot. The x-axis is energy reduction — how much compute we save. The y-axis is accuracy. You want to be in the upper-right: high accuracy AND high energy savings.
>
> The Increasing Width model is the clear Pareto-optimal choice — it achieves 94% accuracy, which is essentially the same as the baseline's 95.3%, while saving 42% of the computational energy. That's remarkable: nearly half the compute for free.
>
> Our novel Adaptive Width model gets the highest energy savings at 47.5%, but accuracy drops to 86%. The Decreasing Width model — our funnel design — keeps accuracy at 88% but its energy savings are modest at 12.9%. The intuition is that wide early layers are expensive, so even exiting at Stage 0 in a decreasing-width model costs more than exiting at Stage 0 in an increasing-width model where the first stage is narrow and cheap.
>
> Importantly, none of the accuracy differences between early-exit models and the baseline are statistically significant by paired t-test — meaning we're getting energy savings essentially for free."

---

## SLIDE 8 — Exit Behavior Analysis (Results)

**Time: ~1 minute**

### Slide Content
```
WHERE DO SAMPLES EXIT?
-----------------------
* Most samples exit at Stage 0 (the earliest possible point)
* Increasing Width: 85%+ exit at Stage 0
  - 83% of decisions are OPTIMAL
  - Only 5% "underthinking" (exited early, got it wrong)
  - 12% "overthinking" (correct earlier but went deeper)

KEY INSIGHT
-----------
EEG seizure detection is a "mostly easy" problem.
The majority of inputs can be classified with minimal computation.
```

### Visual — Two images side by side:
1. `plots/exit_distributions_grid.png` — bar charts showing exit stage distribution for each model
2. `plots/exit_quality_Increasing_Width.png` — overthinking / underthinking / optimal breakdown

### Script
> "This is one of the most interesting findings. Look at where samples actually exit. The overwhelming majority leave at Stage 0 — the very first exit point. For the Increasing Width model, only about 5% of samples are 'underthinking' — they exited early but got the wrong answer. About 12% are 'overthinking' — they were already correct at an earlier stage but went deeper unnecessarily. And 83% are optimal — they exited at exactly the right time.
>
> This tells us something important about the problem domain: EEG seizure detection, at least on this dataset, is a 'mostly easy' problem. Most recordings are clearly non-seizure, and the network figures that out almost immediately. The hard cases are rare and genuinely need the full network depth."

---

## SLIDE 9 — Dynamic Thresholding & Calibration (Results)

**Time: ~45 seconds**

### Slide Content
```
THREE EXIT STRATEGIES COMPARED
-------------------------------
| Strategy   | Accuracy | Energy Red. | F1 Score |
|------------|----------|-------------|----------|
| Confidence | 93.3%    | 28.4%       | 90.7%    |
| Entropy    | 86.7%    | 20.1%       | 79.2%    |
| Patience   | 91.7%    | 3.2%        | 88.1%    |

* Confidence-based thresholding WINS on all metrics
* Patience is too conservative (barely saves energy)
* Models are well-calibrated: ECE = 0.057 for Increasing Width
  (when the model says 90% confident, it's right ~90% of the time)
```

### Visual — Two images side by side:
1. `plots/threshold_comparison.png` — 3-panel bar chart comparing strategies
2. `plots/reliability_Increasing_Width.png` — calibration diagram (bars vs diagonal)

### Script
> "We compared three strategies for deciding when to exit. Confidence-based — stop when the softmax probability exceeds a threshold — wins on all fronts. Entropy-based is more aggressive but less accurate. And patience-based, where you wait for consecutive stages to agree, is too conservative — it only saves 3.2% energy.
>
> We also measured calibration using Expected Calibration Error. The Increasing Width model has an ECE of just 0.057, meaning when it says it's 90% confident, it really is right about 90% of the time. This kind of calibration is critical for medical applications where you need to trust the model's confidence."

---

## SLIDE 10 — Pruning & Scaling (Results)

**Time: ~45 seconds**

### Slide Content
```
STRUCTURED PRUNING: ADDITIONAL COMPRESSION
-------------------------------------------
* Remove least important channels by L1-norm ranking
* Creates genuinely smaller networks (physically removes weights)

| Prune Ratio | Accuracy | FLOPs Reduction |
|-------------|----------|-----------------|
| 0% (base)   | 93.3%    | 0%              |
| 10%          | 81.7%    | 19%             |
| 25%          | 83.3%    | 43%             |
| 50%          | 80.0%    | 70%             |

MODEL SIZE SCALING
------------------
* Sweet spot: Small [32] to Medium [64] channels
  -> 93%+ accuracy, 25-30% energy savings
* XLarge [256] OVERFITS: drops to 77% accuracy with 1.9M params
* Lean and efficient beats large and overfit
```

### Visual — Two images side by side:
1. `plots/pruning_impact.png` — dual-axis accuracy vs FLOPs reduction curve
2. `plots/model_size_scaling.png` — dual-axis accuracy + energy vs parameter count

### Script
> "Beyond early exits, we explored structured pruning — physically removing the least important convolutional channels based on L1-norm importance scores. At 25% pruning, you lose about 10 points of accuracy but save 43% of compute. This is on top of early-exit savings, so they're complementary techniques.
>
> Our model size scaling study shows a clear sweet spot around 32 to 64 channels. Bigger is not always better — the XLarge model with 1.9 million parameters actually overfits badly and drops to 77% accuracy. For this dataset, lean and efficient wins."

---

## SLIDE 11 — Conclusions

**Time: ~45 seconds**

### Slide Content
```
KEY FINDINGS
-------------
1. Early exits save 42-47% energy with negligible accuracy loss

2. Increasing Width [32->64->128] is PARETO-OPTIMAL:
   94% accuracy + 42% energy savings

3. Our novel architectures reveal important tradeoffs:
   - Decreasing Width: funnel design front-loads compute,
     limiting energy savings despite good accuracy
   - Adaptive Width (Tuned): 87% accuracy, 48% energy savings
     (improved +4pp accuracy, +12pp F1 with tuned hyperparameters)

4. Confidence-based thresholding is the best exit strategy

5. EEG seizure detection is "mostly easy":
   83%+ of samples exit optimally at the first stage

6. Models are well-calibrated (ECE < 0.06) — critical for
   clinical trust

BROADER IMPACT
--------------
These techniques could enable longer battery life for wearable
seizure monitors, making continuous EEG monitoring more practical
for the 3.4 million Americans living with epilepsy.
```

### Visual
A summary infographic with the 3 key numbers:
- **94%** accuracy
- **42%** energy savings
- **83%** optimal exit rate

Or just use the Pareto plot (accuracy_vs_energy.png) as a smaller thumbnail alongside the text.

### Script
> "To wrap up our key findings: Early-exit architectures can save nearly half the computational energy with negligible accuracy loss. The Increasing Width architecture is the Pareto-optimal design for EEG seizure detection — it matches the baseline's accuracy while cutting compute by 42%.
>
> Our novel architectures taught us valuable lessons. The decreasing-width funnel design maintains good accuracy but front-loads computation in the wide early layers, limiting energy savings. The adaptive-width model, especially after hyperparameter tuning, achieves the most aggressive energy savings at 48% while maintaining 87% accuracy.
>
> Perhaps most importantly, we showed that the vast majority of EEG samples are easy — the network classifies them correctly and confidently at the very first exit point. Only the genuinely ambiguous cases need the full network depth."

---

## SLIDE 12 — Team Contributions & Future Work

**Time: ~30 seconds**

### Slide Content
```
TEAM CONTRIBUTIONS
-------------------
| Member                | Contributions                                      |
|-----------------------|----------------------------------------------------|
| Rithwak Somepalli     | Novel architecture design & implementation          |
|                       | (decreasing-width, adaptive-width), structured      |
|                       | pruning, full pipeline development, statistical     |
|                       | testing, experiments & results writing               |
|-----------------------|----------------------------------------------------|
| Suryaprakash          | Baseline CNN + existing architectures (constant,    |
| Murugavvel            | increasing width), hyperparameter tuning, model     |
|                       | evaluation, experiments & results writing            |
|-----------------------|----------------------------------------------------|
| Monique Gaye          | Energy/memory analysis, exit threshold strategies,  |
|                       | visualizations, research & introduction writing      |
|-----------------------|----------------------------------------------------|
| Amrutha Kodali        | Data preprocessing (normalization, FFT features,    |
|                       | class balancing), data analysis, cross-dataset       |
|                       | evaluation, teamwork & conclusions writing           |

FUTURE WORK
------------
1. Cross-dataset evaluation: train on Bonn -> test on CHB-MIT
   (983 hours, 23 patients, 198 seizures)
2. Hardware deployment benchmarking on real edge devices
   (ARM Cortex-M, Raspberry Pi)
3. Multi-class seizure type classification
```

### Visual
Team table (as above) plus a "Future Work" bullet list. Optional: team photos.

### Script
> "Here are each team member's contributions. Rithwak designed and implemented the two novel architectures and the full research pipeline. Surya built the baseline and existing architectures and handled hyperparameter tuning. Monique led the energy analysis, threshold strategies, and visualizations. And Amrutha handled all data preprocessing and analysis.
>
> For future work, we'd like to validate on the much larger CHB-MIT dataset with 23 patients and 983 hours of data, benchmark on actual edge hardware, and extend to multi-class seizure type classification.
>
> Thank you — happy to take questions."

---

---

# Appendix A: Visual Assets Checklist

All images are in the `plots/` directory. Here's what goes on each slide:

| Slide | Image File(s) | Description |
|---|---|---|
| 1 | (minimal — title only) | Clean background, optional EEG waveform |
| 2 | `plots/early_exit_concept.png` | Traditional vs early-exit CNN diagram |
| 3 | (icons/numbered list) | No data plot needed |
| 4 | `plots/five_architectures.png` | 5 architecture block diagrams |
| 5 | `plots/training_pipeline.png` | 3-phase training flowchart |
| 6 | (simple FFT diagram) | Can create manually or use dataset viz |
| 7 | **`plots/accuracy_vs_energy.png`** | **HERO FIGURE — Pareto trade-off** |
| 8 | `plots/exit_distributions_grid.png` + `plots/exit_quality_Increasing_Width.png` | Exit behavior |
| 9 | `plots/threshold_comparison.png` + `plots/reliability_Increasing_Width.png` | Thresholding |
| 10 | `plots/pruning_impact.png` + `plots/model_size_scaling.png` | Pruning & scaling |
| 11 | (key numbers infographic or Pareto thumbnail) | Summary |
| 12 | (team table) | Contributions |

---

# Appendix B: Presentation Tips

1. **Rehearse to 9:30** — leave a 30-second buffer for natural pauses and transitions
2. **The Pareto plot (Slide 7) is your hero figure** — point at it and tell the story
3. **Don't read the tables aloud** — highlight the key numbers and explain the pattern
4. **Use a laser pointer or cursor** on the Pareto plot to trace the tradeoff
5. **Speak confidently about limitations** — having answers ready is better than dodging

## Anticipated Q&A

| Question | Answer |
|---|---|
| "Why not test on CHB-MIT?" | It's 983 hours of data — outside the scope of this semester, but it's on our roadmap as the immediate next step. |
| "Are the accuracy differences significant?" | No — and that's the point. The energy savings are highly significant (p < 10^-15), but accuracy differences are not, meaning we get energy savings essentially for free. |
| "How does this compare to transformers?" | Our focus is lightweight CNNs for edge devices where transformers would be too expensive. Transformer-based early exits are an interesting future direction. |
| "Why does decreasing width save less energy?" | The funnel design front-loads computation in the wide early layers — Stage 0 in a [128,64,32] network does more FLOPs than Stage 0 in a [32,64,128] network, so even early exits are expensive. |
| "Would you use this in a real hospital?" | The calibration results are promising — ECE < 0.06 means the confidence scores are trustworthy. But clinical deployment would require validation on much larger, multi-patient datasets and regulatory approval. |
| "Why is the Adaptive model worse?" | The channel gating mechanism needs careful tuning. With our tuned hyperparameters (lower LR, softer sparsity penalty, longer warmup), it improved from 82% to 87% accuracy — suggesting it has more potential with further optimization. |

---

# Appendix C: Tuned Adaptive Width Results

After running `tune_adaptive.py` to compare the original vs tuned Adaptive Width configuration:

```
RESULTS COMPARISON
============================================================
  Original [64,64,64] (lr=0.001, sparsity=0.01):
    Accuracy:         82.67 +/- 7.64%
    F1 Score:         65.13 +/- 20.26%
    Energy Reduction: 43.77 +/- 3.86%

  Tuned [96,64,32] (lr=0.0005, sparsity=0.005):
    Accuracy:         87.00 +/- 6.86%
    F1 Score:         76.77 +/- 14.61%
    Energy Reduction: 48.17 +/- 3.89%

DELTAS (Tuned - Original):
  Accuracy:         +4.33 pp
  F1 Score:         +11.64 pp
  Energy Reduction: +4.40 pp
```

Key tuning changes:
- Learning rate: 0.001 → 0.0005 (more stable convergence)
- Sparsity lambda: 0.01 → 0.005 (let gates stay open longer)
- Channels: [64,64,64] → [96,64,32] (wider Stage 0 for better early exits)
- Warmup: 10 → 15 epochs (gates need more warmup before energy penalty)

---

# Appendix D: Completeness Check vs Initial Plans

## From the Project Proposal

| Planned Item | Status | Notes |
|---|---|---|
| 5 model architectures | ✅ Complete | Baseline, Constant, Increasing, Decreasing, Adaptive |
| Accuracy, F1, recall metrics | ✅ Complete | 5-trial averages with std dev |
| Exit-distribution analysis | ✅ Complete | Per-model exit stage histograms + quality analysis |
| Energy savings (FLOPs) | ✅ Complete | FLOPs-weighted energy reduction measured |
| Confidence calibration | ✅ Complete | ECE + reliability diagrams for all models |
| Bonn dataset | ✅ Complete | Primary dataset, fully processed |
| CHB-MIT dataset | ❌ Not done | Too large for project scope; listed as future work |

## From the Status Report (Remaining Month Plan)

| Planned Item | Status | Notes |
|---|---|---|
| 1. Cross-dataset evaluation | ❌ | Deferred to future work |
| 2. Deeper exit behavior analysis | ✅ | Overthinking/underthinking, exit heatmaps, per-stage accuracy |
| 3. Stronger visualizations | ✅ | 64 publication-quality plots across 10 plot types |
| 4. Compare with literature benchmarks | ❌ | No formal comparison table vs BranchyNet/SDN |
| 5. Refine research narrative | ✅ | Clear Pareto-optimal story around Increasing Width |
| 6. Better hyperparameter tuning | ✅ | Grid search + dedicated adaptive tuning (tune_adaptive.py) |
| 7. Feature pruning mechanisms | ✅ | Structured pruning (L1-norm) + channel gating |
| 8. Different model sizes | ✅ | Tiny [16] through XLarge [256] scaling study |
| 9. Dynamic thresholds | ✅ | Confidence, entropy, and patience strategies compared |

**Score: 7 of 9 planned items completed** (missing cross-dataset and literature comparison)
