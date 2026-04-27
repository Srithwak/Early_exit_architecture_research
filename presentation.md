# Presentation: Early-Exit Neural Network Architectures for Energy-Efficient Biomedical Signal Classification

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
          for Energy-Efficient Biomedical Signal Classification

Authors:  Rithwak Somepalli · Suryaprakash Murugavvel
          Monique Gaye · Amrutha Kodali

Course:   CS [Course Number] — Spring 2026
```

### Visual
Clean layout with a subtle EEG/ECG waveform background element. White background, bold title.

### Script
> "Hi everyone. Today we're presenting our research on early-exit neural network architectures for energy-efficient biomedical signal classification. We explored how to make neural networks smarter about *when to stop computing* — and validated our approach across three clinical datasets."

---

## SLIDE 2 — Problem & Motivation

**Time: ~1 minute**

### Slide Content
```
THE PROBLEM
-----------
* 1 in 26 people will develop epilepsy in their lifetime
* Wearable devices need real-time seizure & arrhythmia detection
* Standard neural networks process every input through ALL layers
  — same compute for obvious cases and ambiguous ones
* This is wasteful for battery-powered medical devices

OUR INSIGHT
-----------
"Not all inputs are equally hard — the network should decide
 when it's confident enough to stop."
```

### Visual
Diagram showing:
- LEFT: Traditional CNN → input flows through all 3 layers every time
- RIGHT: Early-Exit CNN → input can exit after Layer 1 if confident, saving ~42% energy

### Script
> "About 1 in 26 people will develop epilepsy, and millions more have cardiac arrhythmias requiring monitoring. For wearable devices to be practical for continuous use, they need to be extremely energy-efficient. But traditional neural networks process every input through the entire network — whether it's an obvious normal recording or a genuinely ambiguous case.
>
> Our core insight is simple: not all inputs are equally hard. If the network is already confident at an early layer, it should stop computing."

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

3. Cross-dataset evaluation across 3 clinical datasets:
   Bonn EEG (280 samples), ECG Arrhythmia (87K), MIT-BIH (77K)

4. Three dynamic exit strategies:
   Confidence, Entropy, and Patience thresholding

5. Structured pruning for additional model compression
```

### Script
> "Let me highlight what we actually did. First, we proposed two novel architectures — a decreasing-width design and an adaptive-width design with learned channel gates. Second, we ran a fair, bias-controlled comparison across five architectures on three different clinical datasets — from tiny EEG data to large-scale ECG databases. We also compared three exit strategies and explored structured pruning."

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

### Script
> "Here are our five architectures. The baseline is a standard 3-stage CNN with no exits. The constant-width and increasing-width designs come from existing literature.
>
> Our novel contributions are the decreasing-width model — a funnel that captures broad patterns early and narrows down — and the adaptive-width model with learned channel gates that dynamically suppress irrelevant features."

---

## SLIDE 5 — Training Pipeline (Methodology)

**Time: ~1 minute**

### Slide Content
```
TWO-PHASE TRAINING PROTOCOL
------------------------------
Phase 1 - WARMUP (10 epochs on Bonn, 1 on ECG/MIT-BIH)
  * Train classifier heads only
  * Exit policies frozen (lambda = 0)

Phase 2 - JOINT TRAINING (10 epochs on Bonn, 1 on ECG/MIT-BIH)
  * Energy-aware loss function:
    L = SUM[ p_reach * p_exit * L_CE ] + lambda * energy_cost
  * lambda = 0.02 penalizes unnecessary deep computation

Phase 3 - THRESHOLD CALIBRATION
  * Tune exit confidence thresholds on validation set

FAIRNESS: All 5 models get identical total epochs,
          same seeds, same data splits
```

### Script
> "Our training pipeline has two main phases plus calibration. First, a warmup phase training only classifier heads. Then joint training with our energy-aware loss. Finally, we calibrate exit thresholds on a validation set. All five models train under identical conditions for fair comparison."

---

## SLIDE 6 — Datasets (Methodology)

**Time: ~45 seconds**

### Slide Content
```
THREE CLINICAL DATASETS
--------------------------
| Property        | Bonn EEG       | ECG Arrhythmia | MIT-BIH        |
|-----------------|----------------|----------------|----------------|
| Domain          | EEG Seizure    | ECG            | ECG            |
| Train Samples   | 280            | 86,798         | 76,612         |
| Sequence Length  | 4,097          | 187            | 187            |
| Classes         | 2 (binary)     | 5              | 5              |
| Input Channels  | 6              | 6              | 6              |

PREPROCESSING: Z-score normalization + FFT frequency band features
→ 6-channel input (1 raw signal + 5 frequency bands)
```

### Script
> "We evaluated on three datasets spanning two modalities. The Bonn EEG dataset is small but well-studied for seizure detection. The ECG Arrhythmia and MIT-BIH datasets are large-scale 5-class cardiac classification tasks. All use identical preprocessing: Z-score normalization plus FFT frequency band extraction, giving us a 6-channel input."

---

## SLIDE 7 — Main Results: Bonn EEG (KEY SLIDE)

**Time: ~1 minute 15 seconds**

### Slide Content
```
BONN EEG RESULTS (Seizure Detection — Binary, 1 trial)
---------------------------------------------------------
| Model              | Accuracy | F1 Score | Energy Red. |
|--------------------|----------|----------|-------------|
| Baseline CNN       | 75.00%   | 42.86%   | 0%          |
| Constant Width     | 93.33%   | 90.68%   | 29.74%      |
| Increasing Width ⭐ | 93.33%   | 90.68%   | 42.11%      |
| Decreasing Width   | 93.33%   | 90.68%   | 15.83%      |
| Adaptive Width     | 75.00%   | 42.86%   | 48.18%      |

KEY FINDING: Early exits MASSIVELY outperform the baseline
on this small dataset (F1: 42.86% → 90.68%)
```

### Visual — Pareto plot: accuracy vs energy reduction

### Script
> "On the Bonn EEG dataset, the results are striking. The baseline CNN only achieves 75% accuracy and 42.86% F1 — it mostly predicts the majority class. But all three properly-functioning early-exit architectures jump to 93.33% accuracy and 90.68% F1. The joint training with energy-aware loss acts as a powerful regularizer on this tiny dataset.
>
> Among them, Increasing Width is the clear winner: same accuracy but with 42% energy savings and the lowest latency at 1.46ms. Note that Adaptive Width collapsed on this small dataset, matching the baseline's poor performance."

---

## SLIDE 8 — Cross-Dataset Results (KEY SLIDE)

**Time: ~1 minute**

### Slide Content
```
CROSS-DATASET ACCURACY COMPARISON
-----------------------------------
| Model              | Bonn EEG | ECG     | MIT-BIH |
|--------------------|----------|---------|---------|
| Baseline CNN       | 75.00%   | 93.41%  | 94.03%  |
| Constant Width     | 93.33%   | 88.77%  | 91.07%  |
| Increasing Width ⭐ | 93.33%   | 88.94%  | 89.78%  |
| Decreasing Width   | 93.33%   | 88.05%  | 92.06%  |
| Adaptive Width     | 75.00%   | 88.68%  | 89.11%  |

ENERGY REDUCTION (%)
| Increasing Width   | 42.11%   | 25.69%  | 33.60%  |

Note: ECG/MIT-BIH trained only 2 epochs (vs 20 for Bonn)
```

### Script
> "Across all three datasets, Increasing Width consistently achieves the highest energy reduction — 25 to 42% — with the lowest latency. On the larger ECG and MIT-BIH datasets, the baseline actually outperforms early-exit models, but those were trained for only 2 total epochs vs. 20 for Bonn. The early-exit branches need more training to converge properly. The key takeaway is that Increasing Width is the most efficient architecture regardless of dataset."

---

## SLIDE 9 — Dynamic Thresholding & Calibration (Results)

**Time: ~45 seconds**

### Slide Content
```
THREE EXIT STRATEGIES COMPARED (Bonn EEG)
-------------------------------------------
| Strategy   | Accuracy | Energy Red. | F1 Score |
|------------|----------|-------------|----------|
| Confidence | 93.3%    | 28.4%       | 90.7%    |
| Entropy    | 86.7%    | 20.1%       | 79.2%    |
| Patience   | 91.7%    | 3.2%        | 88.1%    |

* Confidence-based thresholding WINS on all metrics
* Patience is too conservative (barely saves energy)

CALIBRATION (ECE — lower is better)
* Bonn: Decreasing Width best (0.026)
* ECG/MIT-BIH: Baseline best (0.013)
* Early exits degrade calibration on large datasets
```

### Script
> "We compared three exit strategies. Confidence-based thresholding wins on all fronts. Patience-based is too conservative, saving only 3.2% energy. For calibration, Decreasing Width has the best ECE on Bonn, but on larger datasets the baseline's calibration is superior — early exits tend to degrade confidence estimates, which is a concern for clinical deployment."

---

## SLIDE 10 — Pruning & Scaling (Results)

**Time: ~45 seconds**

### Slide Content
```
STRUCTURED PRUNING (Bonn EEG)
-------------------------------
| Prune Ratio | Accuracy | FLOPs Reduction |
|-------------|----------|-----------------|
| 0% (base)   | 93.3%    | 0%              |
| 10%          | 81.7%    | 19%             |
| 25%          | 83.3%    | 43%             |
| 50%          | 80.0%    | 70%             |

MODEL SIZE SCALING
------------------
* Sweet spot: Small [32] to Medium [64] channels
* XLarge [256] OVERFITS on small datasets
```

### Script
> "We explored structured pruning — physically removing convolutional channels — as a complementary technique. At 25% pruning, you save 43% FLOPs but lose about 10 points of accuracy. Early exits remain the more graceful efficiency mechanism. Our scaling study confirms the sweet spot is around 32-64 channels — bigger models overfit on small datasets."

---

## SLIDE 11 — Conclusions

**Time: ~45 seconds**

### Slide Content
```
KEY FINDINGS
-------------
1. Early exits save 25-42% energy across all 3 datasets
   Increasing Width [32→64→128] is Pareto-optimal

2. Early exits act as REGULARIZERS on small datasets:
   F1 jumps from 42.86% → 90.68% on Bonn EEG

3. Training budget is critical:
   Under-trained exit branches hurt accuracy on large datasets

4. Novel architectures reveal important tradeoffs:
   - Decreasing Width: best calibration, worst efficiency
   - Adaptive Width: needs more data to avoid gate collapse

5. Confidence-based thresholding is the best exit strategy

6. Calibration degrades with early exits on large datasets
   — a concern for clinical deployment

BROADER IMPACT: Enabling longer battery life for wearable
seizure/arrhythmia monitors
```

### Script
> "To wrap up: early-exit architectures consistently save 25-42% energy, with Increasing Width as the clear Pareto winner. On small datasets, early exits double as regularizers. Our novel architectures revealed that funnel designs have poor efficiency despite good calibration, and adaptive gating needs sufficient data. The key limitation is that early exits degrade calibration on large datasets, which matters for clinical trust."

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
1. Re-run ECG/MIT-BIH with full training budget (10+10 epochs)
2. Hardware deployment benchmarking on edge devices (ARM, RPi)
3. Improve calibration of early-exit models for clinical use
```

### Script
> "Here are each team member's contributions. For future work, our top priority is re-running the ECG and MIT-BIH experiments with full training epochs — we believe this will close the accuracy gap. We'd also like to benchmark on real edge hardware and improve calibration for clinical deployment. Thank you — happy to take questions."

---

# Appendix A: Visual Assets Checklist

| Slide | Image File(s) | Description |
|---|---|---|
| 2 | Early-exit concept diagram | Traditional vs early-exit CNN |
| 4 | 5 architecture block diagrams | Varying widths + gates |
| 5 | Training pipeline flowchart | 2-phase + calibration |
| 6 | Dataset comparison table | 3 datasets side by side |
| 7 | Pareto plot (accuracy vs energy) | Bonn EEG results |
| 8 | Cross-dataset comparison table | All 3 datasets |
| 9 | Threshold comparison bars + reliability diagram | Strategy comparison |
| 10 | Pruning impact + model size scaling | Dual-axis plots |
| 11 | Key numbers infographic | 42% energy, 93.33% acc |

---

# Appendix B: Anticipated Q&A

| Question | Answer |
|---|---|
| "Why do early-exit models perform worse on ECG/MIT-BIH?" | Those models were trained for only 2 total epochs vs 20 for Bonn. The exit branches are under-trained. We expect full training would close the gap. |
| "Are the accuracy differences significant?" | On Bonn with multi-trial runs, energy savings are highly significant (p < 10⁻¹⁵), but accuracy differences between architectures are not — meaning we get energy savings for free. |
| "Why does Adaptive Width fail on Bonn?" | The channel gating mechanism collapses on tiny datasets (280 samples). On the larger ECG/MIT-BIH datasets it works reasonably, suggesting it needs more data. |
| "Why does decreasing width save less energy?" | The funnel design front-loads computation — Stage 0 in a [128,64,32] network costs more FLOPs than Stage 0 in a [32,64,128] network, so even early exits are expensive. |
| "Would you use this in a real hospital?" | The Bonn calibration results are promising (ECE < 0.06), but on larger datasets calibration degrades. Clinical deployment requires better calibration and regulatory approval. |

---

# Appendix C: Full Cross-Dataset Results

See `cross_dataset_analysis.md` for complete per-dataset tables, rankings, and detailed analysis across all three datasets.
