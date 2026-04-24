"""
analysis.py — Deeper exit behavior analysis for early-exit networks.

Provides per-sample, per-class, and per-stage analysis of exit decisions
including overthinking/underthinking detection and difficulty scoring.
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict


def collect_exit_statistics(model, dataloader, threshold_info, device, is_baseline=False):
    """
    Collect detailed per-sample exit statistics.
    
    Returns a DataFrame with columns:
        sample_id, true_label, exit_stage, pred_label, confidence,
        entropy, is_correct, cumulative_flops,
        stage_0_pred, stage_0_conf, stage_0_correct,
        stage_1_pred, stage_1_conf, stage_1_correct,
        ...
    
    This captures ALL stage predictions for every sample, not just the exit stage,
    enabling overthinking/underthinking analysis.
    """
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
            batch_size = x.size(0)

            for b in range(batch_size):
                lbl = y[b].item()
                record = {
                    "sample_id": sample_id,
                    "true_label": lbl,
                }

                # Collect predictions at ALL stages
                cumulative_flops = 0.0
                prev_preds = []
                exit_stage = model.num_stages - 1  # default: last stage
                exit_pred = None
                exit_conf = None
                exit_entropy = None

                for i in range(model.num_stages):
                    # Compute stage cost
                    if gates is not None:
                        active_frac = gates[i][b].mean().item()
                        stage_cost = model.stage_flops[i] * active_frac
                    else:
                        stage_cost = model.stage_flops[i]

                    probs = torch.softmax(logits_list[i][b], dim=-1)
                    conf, pred = torch.max(probs, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                    correct = int(pred.item() == lbl)

                    record[f"stage_{i}_pred"] = pred.item()
                    record[f"stage_{i}_conf"] = conf.item()
                    record[f"stage_{i}_entropy"] = entropy.item()
                    record[f"stage_{i}_correct"] = correct

                    # Track cumulative cost up to this stage
                    cumulative_flops += stage_cost

                    # Determine exit decision
                    if not is_baseline and exit_pred is None:
                        if i < model.num_stages - 1:
                            do_exit, p, c, e = _should_exit(
                                logits_list[i][b], i, threshold_info, prev_preds)
                            prev_preds.append(p)
                            if do_exit:
                                exit_stage = i
                                exit_pred = p
                                exit_conf = c
                                exit_entropy = e
                        else:
                            exit_stage = i
                            exit_pred = pred.item()
                            exit_conf = conf.item()
                            exit_entropy = entropy.item()

                if is_baseline:
                    exit_stage = model.num_stages - 1
                    exit_pred = record[f"stage_{model.num_stages - 1}_pred"]
                    exit_conf = record[f"stage_{model.num_stages - 1}_conf"]
                    exit_entropy = record[f"stage_{model.num_stages - 1}_entropy"]
                    cumulative_flops = sum(model.stage_flops)

                record["exit_stage"] = exit_stage
                record["pred_label"] = exit_pred
                record["confidence"] = exit_conf
                record["entropy"] = exit_entropy
                record["is_correct"] = int(exit_pred == lbl)
                record["cumulative_flops"] = cumulative_flops

                records.append(record)
                sample_id += 1

    return pd.DataFrame(records)


def analyze_exit_patterns(exit_df, num_stages=3):
    """
    Analyze exit patterns from collected statistics.
    
    Returns a dict with:
        - per_class_exits: dict[class] -> list of exit counts per stage
        - per_stage_accuracy: list of accuracy at each stage
        - per_stage_avg_confidence: list of avg confidence at each stage
        - per_stage_avg_entropy: list of avg entropy at each stage
        - overthinking: DataFrame of samples correct at earlier stage but exited later
        - underthinking: DataFrame of samples that exited early but were incorrect
        - confusion_per_stage: dict[stage] -> confusion counts
    """
    results = {}

    # 1. Per-class exit distributions
    per_class_exits = {}
    for cls in sorted(exit_df["true_label"].unique()):
        cls_df = exit_df[exit_df["true_label"] == cls]
        exits = [int((cls_df["exit_stage"] == s).sum()) for s in range(num_stages)]
        per_class_exits[int(cls)] = exits
    results["per_class_exits"] = per_class_exits

    # 2. Per-stage accuracy (accuracy of samples that exited at each stage)
    per_stage_accuracy = []
    per_stage_avg_confidence = []
    per_stage_avg_entropy = []
    per_stage_counts = []

    for s in range(num_stages):
        stage_df = exit_df[exit_df["exit_stage"] == s]
        if len(stage_df) > 0:
            acc = stage_df["is_correct"].mean()
            avg_conf = stage_df["confidence"].mean()
            avg_ent = stage_df["entropy"].mean()
        else:
            acc = 0.0
            avg_conf = 0.0
            avg_ent = 0.0
        per_stage_accuracy.append(float(acc))
        per_stage_avg_confidence.append(float(avg_conf))
        per_stage_avg_entropy.append(float(avg_ent))
        per_stage_counts.append(len(stage_df))

    results["per_stage_accuracy"] = per_stage_accuracy
    results["per_stage_avg_confidence"] = per_stage_avg_confidence
    results["per_stage_avg_entropy"] = per_stage_avg_entropy
    results["per_stage_counts"] = per_stage_counts

    # 3. Overthinking analysis
    # Samples that would have been correct at an earlier stage but exited later
    overthinking_records = []
    for _, row in exit_df.iterrows():
        exit_s = int(row["exit_stage"])
        if exit_s > 0:
            for earlier in range(exit_s):
                if row.get(f"stage_{earlier}_correct", 0) == 1:
                    overthinking_records.append({
                        "sample_id": row["sample_id"],
                        "true_label": row["true_label"],
                        "earliest_correct_stage": earlier,
                        "actual_exit_stage": exit_s,
                        "wasted_stages": exit_s - earlier,
                        "exit_correct": row["is_correct"],
                    })
                    break  # Only record the earliest correct stage

    results["overthinking"] = pd.DataFrame(overthinking_records) if overthinking_records else pd.DataFrame()

    # 4. Underthinking analysis
    # Samples that exited early but were incorrect
    underthinking_records = []
    for _, row in exit_df.iterrows():
        exit_s = int(row["exit_stage"])
        if exit_s < num_stages - 1 and row["is_correct"] == 0:
            # Check if a later stage would have been correct
            later_correct = False
            for later in range(exit_s + 1, num_stages):
                if row.get(f"stage_{later}_correct", 0) == 1:
                    later_correct = True
                    break

            underthinking_records.append({
                "sample_id": row["sample_id"],
                "true_label": row["true_label"],
                "exit_stage": exit_s,
                "confidence_at_exit": row["confidence"],
                "would_be_correct_later": later_correct,
            })

    results["underthinking"] = pd.DataFrame(underthinking_records) if underthinking_records else pd.DataFrame()

    # 5. Summary statistics
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
    """
    Assign a difficulty score to each sample based on how many stages
    it needs for a correct prediction.
    
    Difficulty score:
        0 = correct at stage 0 (easiest)
        1 = correct at stage 1 but not stage 0
        2 = correct at stage 2 but not earlier
        num_stages = never correct (hardest)
    
    Returns a DataFrame with columns: sample_id, true_label, difficulty_score,
        first_correct_stage, num_stages_correct
    """
    model.eval()
    records = []
    sample_id = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            logits_list = outputs["logits"]
            batch_size = x.size(0)

            for b in range(batch_size):
                lbl = y[b].item()
                first_correct = None
                stages_correct = 0

                for i in range(model.num_stages):
                    pred = torch.argmax(logits_list[i][b], dim=-1).item()
                    if pred == lbl:
                        stages_correct += 1
                        if first_correct is None:
                            first_correct = i

                difficulty = first_correct if first_correct is not None else model.num_stages

                records.append({
                    "sample_id": sample_id,
                    "true_label": lbl,
                    "difficulty_score": difficulty,
                    "first_correct_stage": first_correct if first_correct is not None else -1,
                    "num_stages_correct": stages_correct,
                })
                sample_id += 1

    return pd.DataFrame(records)


def print_analysis_report(analysis_results, model_name="Model"):
    """Pretty-print the analysis results."""
    summary = analysis_results["summary"]
    print(f"\n{'='*60}")
    print(f"Exit Behavior Analysis: {model_name}")
    print(f"{'='*60}")
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
    print(f"Average Exit Stage: {summary['avg_exit_stage']:.2f}")
    print(f"Overthinking Rate: {summary['overthinking_rate']:.2%} ({summary['overthinking_count']} samples)")
    print(f"Underthinking Rate: {summary['underthinking_rate']:.2%} ({summary['underthinking_count']} samples)")

    print(f"\nPer-Stage Breakdown:")
    for s, (acc, conf, ent, count) in enumerate(zip(
            analysis_results["per_stage_accuracy"],
            analysis_results["per_stage_avg_confidence"],
            analysis_results["per_stage_avg_entropy"],
            analysis_results["per_stage_counts"])):
        print(f"  Stage {s}: {count:4d} exits | Acc: {acc:.2%} | Avg Conf: {conf:.4f} | Avg Entropy: {ent:.4f}")

    print(f"\nPer-Class Exit Distribution:")
    for cls, exits in analysis_results["per_class_exits"].items():
        total_cls = sum(exits)
        pcts = [f"{e/total_cls*100:.1f}%" if total_cls > 0 else "0%" for e in exits]
        print(f"  Class {cls}: {exits} ({pcts})")

    print(f"{'='*60}\n")
