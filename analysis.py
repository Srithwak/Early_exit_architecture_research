import torch
import numpy as np
import pandas as pd


def collect_exit_statistics(model, dataloader, threshold_info, device, is_baseline=False):
    from evaluate import _should_exit

    model.eval()
    records = []
    sample_id = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            logits_list = outputs["logits"]
            gates = outputs.get("gates", None)

            for b in range(x.size(0)):
                lbl = y[b].item()
                record = {"sample_id": sample_id, "true_label": lbl}

                cumulative_flops = 0.0
                prev_preds = []
                exit_stage = model.num_stages - 1
                exit_pred = exit_conf = exit_entropy = None

                for i in range(model.num_stages):
                    if gates is not None:
                        stage_cost = model.stage_flops[i] * gates[i][b].mean().item()
                    else:
                        stage_cost = model.stage_flops[i]

                    probs = torch.softmax(logits_list[i][b], dim=-1)
                    conf, pred = torch.max(probs, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

                    record[f"stage_{i}_pred"] = pred.item()
                    record[f"stage_{i}_conf"] = conf.item()
                    record[f"stage_{i}_entropy"] = entropy.item()
                    record[f"stage_{i}_correct"] = int(pred.item() == lbl)
                    cumulative_flops += stage_cost

                    if not is_baseline and exit_pred is None:
                        if i < model.num_stages - 1:
                            do_exit, p, c, e = _should_exit(
                                logits_list[i][b], i, threshold_info, prev_preds)
                            prev_preds.append(p)
                            if do_exit:
                                exit_stage, exit_pred, exit_conf, exit_entropy = i, p, c, e
                        else:
                            exit_stage = i
                            exit_pred, exit_conf, exit_entropy = pred.item(), conf.item(), entropy.item()

                if is_baseline:
                    exit_stage = model.num_stages - 1
                    exit_pred = record[f"stage_{model.num_stages - 1}_pred"]
                    exit_conf = record[f"stage_{model.num_stages - 1}_conf"]
                    exit_entropy = record[f"stage_{model.num_stages - 1}_entropy"]
                    cumulative_flops = sum(model.stage_flops)

                record.update({
                    "exit_stage": exit_stage, "pred_label": exit_pred,
                    "confidence": exit_conf, "entropy": exit_entropy,
                    "is_correct": int(exit_pred == lbl), "cumulative_flops": cumulative_flops,
                })
                records.append(record)
                sample_id += 1

    return pd.DataFrame(records)


def analyze_exit_patterns(exit_df, num_stages=3):
    results = {}

    # per-class exit distributions
    per_class_exits = {}
    for cls in sorted(exit_df["true_label"].unique()):
        cls_df = exit_df[exit_df["true_label"] == cls]
        per_class_exits[int(cls)] = [int((cls_df["exit_stage"] == s).sum()) for s in range(num_stages)]
    results["per_class_exits"] = per_class_exits

    # per-stage metrics
    per_stage_accuracy, per_stage_avg_confidence, per_stage_avg_entropy, per_stage_counts = [], [], [], []
    for s in range(num_stages):
        stage_df = exit_df[exit_df["exit_stage"] == s]
        if len(stage_df) > 0:
            per_stage_accuracy.append(float(stage_df["is_correct"].mean()))
            per_stage_avg_confidence.append(float(stage_df["confidence"].mean()))
            per_stage_avg_entropy.append(float(stage_df["entropy"].mean()))
        else:
            per_stage_accuracy.append(0.0)
            per_stage_avg_confidence.append(0.0)
            per_stage_avg_entropy.append(0.0)
        per_stage_counts.append(len(stage_df))

    results["per_stage_accuracy"] = per_stage_accuracy
    results["per_stage_avg_confidence"] = per_stage_avg_confidence
    results["per_stage_avg_entropy"] = per_stage_avg_entropy
    results["per_stage_counts"] = per_stage_counts

    # overthinking: correct at earlier stage but exited later
    overthinking_records = []
    for _, row in exit_df.iterrows():
        exit_s = int(row["exit_stage"])
        if exit_s > 0:
            for earlier in range(exit_s):
                if row.get(f"stage_{earlier}_correct", 0) == 1:
                    overthinking_records.append({
                        "sample_id": row["sample_id"], "true_label": row["true_label"],
                        "earliest_correct_stage": earlier, "actual_exit_stage": exit_s,
                        "wasted_stages": exit_s - earlier, "exit_correct": row["is_correct"],
                    })
                    break
    results["overthinking"] = pd.DataFrame(overthinking_records) if overthinking_records else pd.DataFrame()

    # underthinking: exited early but wrong
    underthinking_records = []
    for _, row in exit_df.iterrows():
        exit_s = int(row["exit_stage"])
        if exit_s < num_stages - 1 and row["is_correct"] == 0:
            later_correct = any(row.get(f"stage_{l}_correct", 0) == 1 for l in range(exit_s + 1, num_stages))
            underthinking_records.append({
                "sample_id": row["sample_id"], "true_label": row["true_label"],
                "exit_stage": exit_s, "confidence_at_exit": row["confidence"],
                "would_be_correct_later": later_correct,
            })
    results["underthinking"] = pd.DataFrame(underthinking_records) if underthinking_records else pd.DataFrame()

    total = len(exit_df)
    results["summary"] = {
        "total_samples": total,
        "overall_accuracy": float(exit_df["is_correct"].mean()),
        "avg_exit_stage": float(exit_df["exit_stage"].mean()),
        "overthinking_count": len(results["overthinking"]),
        "overthinking_rate": len(results["overthinking"]) / total if total > 0 else 0,
        "underthinking_count": len(results["underthinking"]),
        "underthinking_rate": len(results["underthinking"]) / total if total > 0 else 0,
    }
    return results


def compute_difficulty_scores(model, dataloader, device):
    model.eval()
    records = []
    sample_id = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            logits_list = outputs["logits"]

            for b in range(x.size(0)):
                lbl = y[b].item()
                first_correct = None
                stages_correct = 0
                for i in range(model.num_stages):
                    if torch.argmax(logits_list[i][b], dim=-1).item() == lbl:
                        stages_correct += 1
                        if first_correct is None:
                            first_correct = i

                records.append({
                    "sample_id": sample_id, "true_label": lbl,
                    "difficulty_score": first_correct if first_correct is not None else model.num_stages,
                    "first_correct_stage": first_correct if first_correct is not None else -1,
                    "num_stages_correct": stages_correct,
                })
                sample_id += 1

    return pd.DataFrame(records)


def print_analysis_report(analysis_results, model_name="Model"):
    s = analysis_results["summary"]
    print(f"\n  {model_name}: acc={s['overall_accuracy']:.2%}, avg_exit={s['avg_exit_stage']:.2f}, "
          f"overthink={s['overthinking_rate']:.1%}, underthink={s['underthinking_rate']:.1%}")
    for i, (acc, count) in enumerate(zip(analysis_results["per_stage_accuracy"], analysis_results["per_stage_counts"])):
        print(f"    Stage {i}: {count} exits, acc={acc:.2%}")
