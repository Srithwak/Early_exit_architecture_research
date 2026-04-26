import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

def inspect_data(train_dl):
    """Display preprocessed data attributes for sanity checking."""
    print("\n--- Preprocessed Data Attributes ---")
    X_sample, y_sample = next(iter(train_dl))
    print(f"1. Structured Organization: Batch Shape (X) is {X_sample.shape} -> (Batch Size, Channels, Sequence Length)")
    print(f"   Batch Shape (y): {y_sample.shape}")

    has_nans = torch.isnan(X_sample).any().item()
    print(f"2. Cleanliness: Contains NaNs? {'Yes' if has_nans else 'No'}")

    mean_val = X_sample.mean().item()
    std_val = X_sample.std().item()
    print(f"3. Normalization: Mean ~= {mean_val:.4f}, Std Dev ~= {std_val:.4f} (Z-score normalized)")

    y_all = train_dl.dataset.y
    counts = np.bincount(y_all)
    print("4. Class Imbalance (Training Set):")
    for i, count in enumerate(counts):
        print(f"   Class {i}: {count} samples ({count/len(y_all)*100:.1f}%)")

    print("5. Reduced Complexity: Sequences are preprocessed and formatted specifically for Conv1D.")
    print("------------------------------------\n")

# ──────────────────────────────────────────────
# Exit Decision Logic (strategy-aware)
# ──────────────────────────────────────────────

def _should_exit(logits, stage_idx, threshold_info, prev_preds=None):
    """
    Determine whether a sample should exit at the given stage.
    
    Args:
        logits: (num_classes,) logits for this sample at this stage
        stage_idx: current stage index
        threshold_info: dict from calibrate_thresholds with strategy and thresholds
        prev_preds: list of previous stage predictions (for patience strategy)
    
    Returns:
        (should_exit: bool, pred: int, confidence: float, entropy: float)
    """
    probs = torch.softmax(logits, dim=-1)
    conf, pred = torch.max(probs, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    strategy = threshold_info["strategy"]
    thresholds = threshold_info.get("thresholds", [])

    if strategy == "confidence":
        if stage_idx < len(thresholds):
            do_exit = conf.item() > thresholds[stage_idx]
        else:
            do_exit = True  # Last stage always exits
        return do_exit, pred.item(), conf.item(), entropy.item()

    elif strategy == "entropy":
        if stage_idx < len(thresholds):
            do_exit = entropy.item() < thresholds[stage_idx]
        else:
            do_exit = True
        return do_exit, pred.item(), conf.item(), entropy.item()

    elif strategy == "patience":
        patience_count = threshold_info.get("patience", 2)
        if prev_preds is not None and len(prev_preds) >= patience_count - 1:
            # Check if the last (patience_count - 1) predictions match current
            recent = prev_preds[-(patience_count - 1):]
            do_exit = all(p == pred.item() for p in recent)
        else:
            do_exit = False
        return do_exit, pred.item(), conf.item(), entropy.item()

    else:
        # Fallback: confidence-based
        do_exit = conf.item() > 0.95
        return do_exit, pred.item(), conf.item(), entropy.item()


# ──────────────────────────────────────────────
# Basic Evaluation (backward-compatible)
# ──────────────────────────────────────────────

def evaluate_model(model, test_dl, thresholds, device, is_baseline=False):
    """
    Basic evaluation function. Accepts thresholds as either:
    - A plain list (backward compat with old calibrate_thresholds)
    - A dict from the new calibrate_thresholds (strategy-aware)
    """
    # Normalize threshold input
    if isinstance(thresholds, list):
        threshold_info = {"strategy": "confidence", "thresholds": thresholds}
    else:
        threshold_info = thresholds

    model.eval()
    stage_exits = [0] * model.num_stages
    class_exits = {0: [0] * model.num_stages, 1: [0] * model.num_stages}
    total_samples = 0
    energy_consumed = 0.0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            logits_list = outputs["logits"]
            gates = outputs.get("gates", None)
            batch_size = x.size(0)

            for b in range(batch_size):
                total_samples += 1
                lbl = y[b].item()
                all_targets.append(lbl)

                if is_baseline:
                    energy_consumed += sum(model.stage_flops)
                    stage_exits[-1] += 1
                    class_exits[lbl][-1] += 1
                    pred = torch.argmax(logits_list[-1][b], dim=-1)
                    all_preds.append(pred.item())
                else:
                    prev_preds = []
                    for i in range(model.num_stages):
                        if gates is not None:
                            active_frac = gates[i][b].mean().item()
                            energy_consumed += model.stage_flops[i] * active_frac
                        else:
                            energy_consumed += model.stage_flops[i]

                        if i < model.num_stages - 1:
                            do_exit, pred, conf, ent = _should_exit(
                                logits_list[i][b], i, threshold_info, prev_preds)
                            prev_preds.append(pred)

                            if do_exit:
                                stage_exits[i] += 1
                                class_exits[lbl][i] += 1
                                all_preds.append(pred)
                                break
                        else:
                            stage_exits[i] += 1
                            class_exits[lbl][i] += 1
                            pred = torch.argmax(logits_list[i][b], dim=-1)
                            all_preds.append(pred.item())
                            break

    acc = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    full_energy = sum(model.stage_flops)
    actual_energy = energy_consumed / total_samples
    energy_red = 1.0 - (actual_energy / full_energy)

    return acc, recall, f1, energy_red, stage_exits, class_exits


# ──────────────────────────────────────────────
# ECE Calculation
# ──────────────────────────────────────────────

def calculate_ece(confidences, predictions, targets, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).
    
    Uses equal-width binning with n_bins bins.
    Returns both the scalar ECE and per-bin data for reliability diagrams.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_data.append({
                "bin_lower": bin_lower, "bin_upper": bin_upper,
                "accuracy": float(accuracy_in_bin),
                "confidence": float(avg_confidence_in_bin),
                "count": int(in_bin.sum()),
                "proportion": float(prop_in_bin)
            })
        else:
            bin_data.append({
                "bin_lower": bin_lower, "bin_upper": bin_upper,
                "accuracy": 0.0, "confidence": 0.0,
                "count": 0, "proportion": 0.0
            })

    return float(ece), bin_data


# ──────────────────────────────────────────────
# Advanced Evaluation (strategy-aware)
# ──────────────────────────────────────────────

def evaluate_model_advanced(model, test_dl, thresholds, device, is_baseline=False):
    """
    Advanced evaluation with ECE, latency, and per-sample exit tracking.
    
    Accepts thresholds as either a plain list or a strategy dict.
    
    Returns:
        acc, recall, f1, ece, energy_red, avg_latency, per_sample_data
    
    per_sample_data is a list of dicts with keys:
        sample_id, true_label, pred_label, exit_stage, confidence,
        entropy, is_correct, cumulative_flops
    """
    # Normalize threshold input
    if isinstance(thresholds, list):
        threshold_info = {"strategy": "confidence", "thresholds": thresholds}
    else:
        threshold_info = thresholds

    model.eval()
    total_samples = 0
    energy_consumed = 0.0
    total_latency = 0.0

    all_targets, all_preds, all_confs = [], [], []
    per_sample_data = []

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)

            # Measure Latency
            start_time = time.time()
            outputs = model(x)
            batch_latency = (time.time() - start_time) * 1000  # ms
            total_latency += batch_latency

            logits_list = outputs["logits"]
            gates = outputs.get("gates", None)
            batch_size = x.size(0)

            for b in range(batch_size):
                lbl = y[b].item()
                all_targets.append(lbl)
                sample_energy = 0.0

                if is_baseline:
                    energy_consumed += sum(model.stage_flops)
                    sample_energy = sum(model.stage_flops)
                    probs = torch.softmax(logits_list[-1][b], dim=-1)
                    conf, pred = torch.max(probs, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                    all_preds.append(pred.item())
                    all_confs.append(conf.item())
                    per_sample_data.append({
                        "sample_id": total_samples,
                        "true_label": lbl,
                        "pred_label": pred.item(),
                        "exit_stage": model.num_stages - 1,
                        "confidence": conf.item(),
                        "entropy": entropy.item(),
                        "is_correct": int(pred.item() == lbl),
                        "cumulative_flops": sample_energy
                    })
                else:
                    prev_preds = []
                    exited = False
                    for i in range(model.num_stages):
                        if gates is not None:
                            active_frac = gates[i][b].mean().item()
                            stage_cost = model.stage_flops[i] * active_frac
                        else:
                            stage_cost = model.stage_flops[i]
                        energy_consumed += stage_cost
                        sample_energy += stage_cost

                        if i < model.num_stages - 1:
                            do_exit, pred, conf, ent = _should_exit(
                                logits_list[i][b], i, threshold_info, prev_preds)
                            prev_preds.append(pred)

                            if do_exit:
                                all_preds.append(pred)
                                all_confs.append(conf)
                                per_sample_data.append({
                                    "sample_id": total_samples,
                                    "true_label": lbl,
                                    "pred_label": pred,
                                    "exit_stage": i,
                                    "confidence": conf,
                                    "entropy": ent,
                                    "is_correct": int(pred == lbl),
                                    "cumulative_flops": sample_energy
                                })
                                exited = True
                                break
                        else:
                            # Last stage — must exit
                            probs = torch.softmax(logits_list[i][b], dim=-1)
                            conf_val, pred_val = torch.max(probs, dim=-1)
                            entropy_val = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                            all_preds.append(pred_val.item())
                            all_confs.append(conf_val.item())
                            per_sample_data.append({
                                "sample_id": total_samples,
                                "true_label": lbl,
                                "pred_label": pred_val.item(),
                                "exit_stage": i,
                                "confidence": conf_val.item(),
                                "entropy": entropy_val.item(),
                                "is_correct": int(pred_val.item() == lbl),
                                "cumulative_flops": sample_energy
                            })
                            exited = True
                            break

                total_samples += 1

    acc = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    ece, ece_bin_data = calculate_ece(np.array(all_confs), np.array(all_preds), np.array(all_targets))

    full_energy = sum(model.stage_flops)
    actual_energy = energy_consumed / total_samples
    energy_red = 1.0 - (actual_energy / full_energy)
    avg_latency = total_latency / total_samples

    return acc, recall, f1, ece, energy_red, avg_latency, per_sample_data, ece_bin_data
