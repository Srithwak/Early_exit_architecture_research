import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score


def inspect_data(train_dl):
    X_sample, y_sample = next(iter(train_dl))
    print(f"\nData: X={X_sample.shape}, y={y_sample.shape}")
    print(f"  Mean={X_sample.mean():.4f}, Std={X_sample.std():.4f}, NaN={torch.isnan(X_sample).any().item()}")
    y_all = train_dl.dataset.y
    counts = np.bincount(y_all)
    for i, c in enumerate(counts):
        print(f"  Class {i}: {c} ({c/len(y_all)*100:.1f}%)")
    print()


def _should_exit(logits, stage_idx, threshold_info, prev_preds=None):
    probs = torch.softmax(logits, dim=-1)
    conf, pred = torch.max(probs, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    strategy = threshold_info["strategy"]
    thresholds = threshold_info.get("thresholds", [])

    if strategy == "confidence":
        do_exit = conf.item() > thresholds[stage_idx] if stage_idx < len(thresholds) else True
    elif strategy == "entropy":
        do_exit = entropy.item() < thresholds[stage_idx] if stage_idx < len(thresholds) else True
    elif strategy == "patience":
        patience_count = threshold_info.get("patience", 2)
        if prev_preds is not None and len(prev_preds) >= patience_count - 1:
            recent = prev_preds[-(patience_count - 1):]
            do_exit = all(p == pred.item() for p in recent)
        else:
            do_exit = False
    else:
        do_exit = conf.item() > 0.95

    return do_exit, pred.item(), conf.item(), entropy.item()


def evaluate_model(model, test_dl, thresholds, device, is_baseline=False):
    if isinstance(thresholds, list):
        threshold_info = {"strategy": "confidence", "thresholds": thresholds}
    else:
        threshold_info = thresholds

    model.eval()
    stage_exits = [0] * model.num_stages
    class_exits = {0: [0] * model.num_stages, 1: [0] * model.num_stages}
    total_samples = 0
    energy_consumed = 0.0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            logits_list = outputs["logits"]
            gates = outputs.get("gates", None)

            for b in range(x.size(0)):
                total_samples += 1
                lbl = y[b].item()
                all_targets.append(lbl)

                if is_baseline:
                    energy_consumed += sum(model.stage_flops)
                    stage_exits[-1] += 1
                    class_exits[lbl][-1] += 1
                    all_preds.append(torch.argmax(logits_list[-1][b], dim=-1).item())
                else:
                    prev_preds = []
                    for i in range(model.num_stages):
                        if gates is not None:
                            energy_consumed += model.stage_flops[i] * gates[i][b].mean().item()
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
                            all_preds.append(torch.argmax(logits_list[i][b], dim=-1).item())

    acc = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    full_energy = sum(model.stage_flops)
    energy_red = 1.0 - (energy_consumed / total_samples / full_energy)

    return acc, recall, f1, energy_red, stage_exits, class_exits


def calculate_ece(confidences, predictions, targets, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for bl, bu in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = np.logical_and(confidences > bl, confidences <= bu)
        prop = in_bin.mean()
        if prop > 0:
            acc_bin = (predictions[in_bin] == targets[in_bin]).mean()
            conf_bin = confidences[in_bin].mean()
            ece += np.abs(conf_bin - acc_bin) * prop
            bin_data.append({"bin_lower": bl, "bin_upper": bu, "accuracy": float(acc_bin),
                           "confidence": float(conf_bin), "count": int(in_bin.sum()), "proportion": float(prop)})
        else:
            bin_data.append({"bin_lower": bl, "bin_upper": bu, "accuracy": 0.0,
                           "confidence": 0.0, "count": 0, "proportion": 0.0})

    return float(ece), bin_data


def evaluate_model_advanced(model, test_dl, thresholds, device, is_baseline=False):
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

            start_time = time.time()
            outputs = model(x)
            total_latency += (time.time() - start_time) * 1000

            logits_list = outputs["logits"]
            gates = outputs.get("gates", None)

            for b in range(x.size(0)):
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
                        "sample_id": total_samples, "true_label": lbl,
                        "pred_label": pred.item(), "exit_stage": model.num_stages - 1,
                        "confidence": conf.item(), "entropy": entropy.item(),
                        "is_correct": int(pred.item() == lbl), "cumulative_flops": sample_energy
                    })
                else:
                    prev_preds = []
                    for i in range(model.num_stages):
                        stage_cost = model.stage_flops[i]
                        if gates is not None:
                            stage_cost *= gates[i][b].mean().item()
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
                                    "sample_id": total_samples, "true_label": lbl,
                                    "pred_label": pred, "exit_stage": i,
                                    "confidence": conf, "entropy": ent,
                                    "is_correct": int(pred == lbl), "cumulative_flops": sample_energy
                                })
                                break
                        else:
                            probs = torch.softmax(logits_list[i][b], dim=-1)
                            conf_val, pred_val = torch.max(probs, dim=-1)
                            entropy_val = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                            all_preds.append(pred_val.item())
                            all_confs.append(conf_val.item())
                            per_sample_data.append({
                                "sample_id": total_samples, "true_label": lbl,
                                "pred_label": pred_val.item(), "exit_stage": i,
                                "confidence": conf_val.item(), "entropy": entropy_val.item(),
                                "is_correct": int(pred_val.item() == lbl), "cumulative_flops": sample_energy
                            })

                total_samples += 1

    acc = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    ece, ece_bin_data = calculate_ece(np.array(all_confs), np.array(all_preds), np.array(all_targets))

    full_energy = sum(model.stage_flops)
    energy_red = 1.0 - (energy_consumed / total_samples / full_energy)
    avg_latency = total_latency / total_samples

    return acc, recall, f1, ece, energy_red, avg_latency, per_sample_data, ece_bin_data
