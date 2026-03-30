import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score

def calculate_ece(confidences, predictions, targets, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

def evaluate_model(model, test_dl, thresholds, device, is_baseline=False):
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
                    for i in range(model.num_stages):
                        if gates is not None:
                            active_frac = gates[i][b].mean().item()
                            energy_consumed += model.stage_flops[i] * active_frac
                        else:
                            energy_consumed += model.stage_flops[i]

                        if i < model.num_stages - 1:
                            probs = torch.softmax(logits_list[i][b], dim=-1)
                            conf, pred = torch.max(probs, dim=-1)

                            if conf.item() > thresholds[i]:
                                stage_exits[i] += 1
                                class_exits[lbl][i] += 1
                                all_preds.append(pred.item())
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

def evaluate_model_advanced(model, test_dl, thresholds, device, is_baseline=False):
    model.eval()
    total_samples = 0
    energy_consumed = 0.0
    total_latency = 0.0

    all_targets, all_preds, all_confs = [], [], []

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)

            # Measure Latency
            start_time = time.time()
            outputs = model(x)
            batch_latency = (time.time() - start_time) * 1000 # ms
            total_latency += batch_latency

            logits_list = outputs["logits"]
            gates = outputs.get("gates", None)
            batch_size = x.size(0)

            for b in range(batch_size):
                total_samples += 1
                lbl = y[b].item()
                all_targets.append(lbl)

                if is_baseline:
                    energy_consumed += sum(model.stage_flops)
                    probs = torch.softmax(logits_list[-1][b], dim=-1)
                    conf, pred = torch.max(probs, dim=-1)
                    all_preds.append(pred.item())
                    all_confs.append(conf.item())
                else:
                    for i in range(model.num_stages):
                        if gates is not None:
                            active_frac = gates[i][b].mean().item()
                            energy_consumed += model.stage_flops[i] * active_frac
                        else:
                            energy_consumed += model.stage_flops[i]

                        probs = torch.softmax(logits_list[i][b], dim=-1)
                        conf, pred = torch.max(probs, dim=-1)

                        if i < model.num_stages - 1 and conf.item() > thresholds[i]:
                            all_preds.append(pred.item())
                            all_confs.append(conf.item())
                            break
                        elif i == model.num_stages - 1:
                            all_preds.append(pred.item())
                            all_confs.append(conf.item())
                            break

    acc = accuracy_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    ece = calculate_ece(np.array(all_confs), np.array(all_preds), np.array(all_targets))

    full_energy = sum(model.stage_flops)
    actual_energy = energy_consumed / total_samples
    energy_red = 1.0 - (actual_energy / full_energy)
    avg_latency = total_latency / total_samples

    return acc, recall, f1, ece, energy_red, avg_latency
