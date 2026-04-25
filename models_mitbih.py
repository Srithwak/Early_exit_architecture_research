"""
models_mitbih.py — Model architectures adapted for short-sequence datasets (MIT-BIH).

The MIT-BIH heartbeat sequences are only 187 samples long, so the standard
Conv1dBlock (stride-2 conv + pool + conv + pool = /16 per block) would collapse
the spatial dimension too quickly. This module provides lighter blocks that
use /4 per stage instead of /16.
"""

import torch
import torch.nn as nn
from models import (
    PolicyHead, ClassifierHead, ChannelGate,
    EnergyJointLoss, AdaptiveEnergyJointLoss,
    compute_channel_importance,
)


# ──────────────────────────────────────────────
# Lighter Conv Block for Short Sequences
# ──────────────────────────────────────────────

class Conv1dBlockSmall(nn.Module):
    """
    A lighter convolutional block for short sequences (e.g. 187).
    Uses stride-1 convolutions with a single MaxPool(2) for /2 downsampling,
    plus one additional pool => total /4 per block.

    Size flow: input_len -> /2 (stride-2 conv) -> /2 (pool) = /4 total
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# Generic Early Exit Network (Short Sequences)
# ──────────────────────────────────────────────

class GenericEarlyExitNetSmall(nn.Module):
    """
    Early-exit network for short sequences (e.g. MIT-BIH, 187 samples).
    Uses Conv1dBlockSmall (/4 per stage) instead of Conv1dBlock (/16).

    Size flow for 187:
      Entry: 187 -> 93 (stride-2) -> 46 (pool) = 46
      Stage 1 (Identity): 46
      Stage 2: 46 -> 23 (stride-2) -> 11 (pool) = 11
      Stage 3: 11 -> 5 (stride-2) -> 2 (pool) = 2
    All stages produce valid spatial dims.
    """
    def __init__(self, in_channels=6, channel_sizes=[64, 64, 64], num_classes=5, seq_len=187):
        super().__init__()
        self.num_stages = len(channel_sizes)
        self.channel_progression = channel_sizes
        self.stage_flops = []

        # Entry block
        self.entry = Conv1dBlockSmall(in_channels, channel_sizes[0])
        self.stages = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.policies = nn.ModuleList()

        curr_seq = seq_len
        for i in range(self.num_stages):
            if i == 0:
                self.stages.append(nn.Identity())
                in_ch = in_channels
                out_ch = channel_sizes[0]
            else:
                self.stages.append(Conv1dBlockSmall(channel_sizes[i - 1], channel_sizes[i]))
                in_ch = channel_sizes[i - 1]
                out_ch = channel_sizes[i]

            # FLOP calculation (approximate)
            seq_1 = max(curr_seq // 2, 1)
            flops_conv1 = seq_1 * 5 * in_ch * out_ch
            flops_conv2 = seq_1 * 3 * out_ch * out_ch
            self.stage_flops.append(float(flops_conv1 + flops_conv2))
            curr_seq = max(seq_1 // 2, 1)

            self.classifiers.append(ClassifierHead(channel_sizes[i], num_classes))

            if i < self.num_stages - 1:
                self.policies.append(PolicyHead(channel_sizes[i]))

    def forward(self, x):
        x = self.entry(x)
        logits_list, p_exits_list = [], []

        for i in range(self.num_stages):
            x = self.stages[i](x)
            logits_list.append(self.classifiers[i](x))

            if i < self.num_stages - 1:
                p_exits_list.append(self.policies[i](x))

        return {"logits": logits_list, "p_exits": p_exits_list}

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Adaptive Early Exit Network (Short Sequences)
# ──────────────────────────────────────────────

class AdaptiveEarlyExitNetSmall(GenericEarlyExitNetSmall):
    """Adaptive (channel-gated) version for short sequences."""
    def __init__(self, in_channels=6, channel_sizes=[64, 64, 64], num_classes=5, seq_len=187):
        super().__init__(in_channels, channel_sizes, num_classes, seq_len)
        self.gates = nn.ModuleList()
        for ch in channel_sizes:
            self.gates.append(ChannelGate(ch))

    def forward(self, x):
        x = self.entry(x)
        logits_list, p_exits_list, gate_scores_list = [], [], []

        for i in range(self.num_stages):
            x = self.stages[i](x)
            x, gate_score = self.gates[i](x)
            gate_scores_list.append(gate_score)

            logits_list.append(self.classifiers[i](x))
            if i < self.num_stages - 1:
                p_exits_list.append(self.policies[i](x))

        return {"logits": logits_list, "p_exits": p_exits_list, "gates": gate_scores_list}


# ──────────────────────────────────────────────
# Structured Pruning (Short Sequence version)
# ──────────────────────────────────────────────

def apply_structured_pruning_small(model, prune_ratio=0.25, in_channels=6, seq_len=187, num_classes=5):
    """
    Apply structured pruning for the small model architecture.
    Creates a new, smaller network with genuinely reduced channel counts.
    """
    importance = compute_channel_importance(model)
    original_channels = model.channel_progression

    # Determine pruned channel sizes
    pruned_channels = []
    for i, ch in enumerate(original_channels):
        pruned_ch = max(4, int(ch * (1.0 - prune_ratio)))
        pruned_channels.append(pruned_ch)

    print(f"[Pruning] Original channels: {original_channels} -> Pruned channels: {pruned_channels}")

    # Create new model with pruned architecture
    is_adaptive = isinstance(model, AdaptiveEarlyExitNetSmall)
    if is_adaptive:
        pruned_model = AdaptiveEarlyExitNetSmall(
            in_channels=in_channels, channel_sizes=pruned_channels,
            num_classes=num_classes, seq_len=seq_len
        )
    else:
        pruned_model = GenericEarlyExitNetSmall(
            in_channels=in_channels, channel_sizes=pruned_channels,
            num_classes=num_classes, seq_len=seq_len
        )

    # Transfer surviving weights (reuse logic from models.py)
    _transfer_pruned_weights_small(model, pruned_model, importance, original_channels, pruned_channels)

    param_reduction = 1.0 - pruned_model.count_parameters() / model.count_parameters()
    flops_reduction = 1.0 - sum(pruned_model.stage_flops) / sum(model.stage_flops)
    print(f"[Pruning] Parameter reduction: {param_reduction:.1%}")
    print(f"[Pruning] FLOPs reduction: {flops_reduction:.1%}")

    return pruned_model, pruned_channels


def _transfer_pruned_weights_small(original, pruned, importance, orig_channels, pruned_channels):
    """Transfer weights from original to pruned small model."""
    orig_conv_layers = [(n, m) for n, m in original.named_modules() if isinstance(m, nn.Conv1d)]
    pruned_conv_layers = [(n, m) for n, m in pruned.named_modules() if isinstance(m, nn.Conv1d)]

    for (orig_name, orig_mod), (pruned_name, pruned_mod) in zip(orig_conv_layers, pruned_conv_layers):
        pruned_out = pruned_mod.out_channels
        pruned_in = pruned_mod.in_channels

        if orig_name in importance:
            scores = importance[orig_name]
            _, keep_idx = torch.topk(scores, min(pruned_out, len(scores)))
            keep_idx, _ = keep_idx.sort()
        else:
            keep_idx = torch.arange(pruned_out)

        with torch.no_grad():
            w = orig_mod.weight.data[keep_idx][:, :pruned_in, :]
            pruned_mod.weight.data.copy_(w)
            if orig_mod.bias is not None and pruned_mod.bias is not None:
                pruned_mod.bias.data.copy_(orig_mod.bias.data[keep_idx])

    # Transfer BatchNorm
    orig_bn = [(n, m) for n, m in original.named_modules() if isinstance(m, nn.BatchNorm1d)]
    pruned_bn = [(n, m) for n, m in pruned.named_modules() if isinstance(m, nn.BatchNorm1d)]

    for (_, ob), (_, pb) in zip(orig_bn, pruned_bn):
        n = pb.num_features
        with torch.no_grad():
            pb.weight.data.copy_(ob.weight.data[:n])
            pb.bias.data.copy_(ob.bias.data[:n])
            pb.running_mean.copy_(ob.running_mean[:n])
            pb.running_var.copy_(ob.running_var[:n])
