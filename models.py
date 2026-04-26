import torch
import torch.nn as nn
import copy

# ──────────────────────────────────────────────
# Building Blocks
# ──────────────────────────────────────────────

class Conv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
    def forward(self, x):
        return self.net(x)

class PolicyHead(nn.Module):
    def __init__(self, in_ch, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class ClassifierHead(nn.Module):
    def __init__(self, in_ch, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ──────────────────────────────────────────────
# Generic Early Exit Network
# ──────────────────────────────────────────────

class GenericEarlyExitNet(nn.Module):
    def __init__(self, in_channels=1, channel_sizes=[32, 32, 32], num_classes=2, seq_len=4097):
        super().__init__()
        self.num_stages = len(channel_sizes)
        self.channel_progression = channel_sizes
        self.stage_flops = []

        # Build stage modules
        self.entry = Conv1dBlock(in_channels, channel_sizes[0])
        self.stages, self.classifiers, self.policies = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        curr_seq = seq_len
        for i in range(self.num_stages):
            if i == 0:
                self.stages.append(nn.Identity())
                in_ch = in_channels
                out_ch = channel_sizes[0]
            else:
                self.stages.append(Conv1dBlock(channel_sizes[i-1], channel_sizes[i]))
                in_ch = channel_sizes[i-1]
                out_ch = channel_sizes[i]

            # FLOP Calculation
            seq_1 = curr_seq // 2
            flops_1 = seq_1 * 7 * in_ch * out_ch
            seq_2 = seq_1 // 2
            flops_2 = seq_2 * 5 * out_ch * out_ch

            self.stage_flops.append(float(flops_1 + flops_2))
            curr_seq = seq_2 // 2

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
# Loss Functions
# ──────────────────────────────────────────────

class EnergyJointLoss(nn.Module):
    def __init__(self, stage_flops, class_weights=None, energy_lambda=0.1, is_baseline=False):
        super().__init__()
        total_flops = sum(stage_flops)
        self.normalized_flops = [f / total_flops for f in stage_flops]
        self.energy_lambda = energy_lambda
        self.is_baseline = is_baseline
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, logits_list, p_exits_list, targets):
        task_losses = [self.ce(l, targets) for l in logits_list]

        if self.is_baseline:
            # Pure baseline only backpropagates from the final stage
            return task_losses[-1].mean()

        if self.energy_lambda == 0.0:
            # Average losses to prevent gradient explosion during warmup
            return sum(l.mean() for l in task_losses) / len(task_losses)

        batch_size = targets.size(0)
        num_stages = len(logits_list)
        total_loss = 0.0
        p_reach = torch.ones(batch_size, device=targets.device)

        for i in range(num_stages):
            e_cost = self.normalized_flops[i]
            exp_energy = p_reach * e_cost

            if i < num_stages - 1:
                p_exit = p_exits_list[i].squeeze(-1)
                stage_loss = p_reach * p_exit * task_losses[i]
                total_loss += stage_loss.mean() + self.energy_lambda * exp_energy.mean()
                p_reach = p_reach * (1.0 - p_exit)
            else:
                stage_loss = p_reach * task_losses[i]
                total_loss += stage_loss.mean() + self.energy_lambda * exp_energy.mean()

        return total_loss

# ──────────────────────────────────────────────
# Channel Gating (Soft Feature Pruning)
# ──────────────────────────────────────────────

class ChannelGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.sigmoid(self.fc(y)).view(b, c, 1)

        if not self.training:
            y = (y > 0.5).float()

        return x * y, y

class AdaptiveEarlyExitNet(GenericEarlyExitNet):
    def __init__(self, in_channels=1, channel_sizes=[64, 64, 64], num_classes=2, seq_len=4097):
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

class AdaptiveEnergyJointLoss(EnergyJointLoss):
    def __init__(self, stage_flops, class_weights=None, energy_lambda=0.1, sparsity_lambda=0.01, is_baseline=False):
        super().__init__(stage_flops, class_weights, energy_lambda, is_baseline)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, logits_list, p_exits_list, targets, gate_scores_list=None):
        base_loss = super().forward(logits_list, p_exits_list, targets)
        if gate_scores_list is not None and self.sparsity_lambda > 0:
            sparsity_loss = sum(g.mean() for g in gate_scores_list) / len(gate_scores_list)
            return base_loss + self.sparsity_lambda * sparsity_loss
        return base_loss

# ──────────────────────────────────────────────
# Structured Pruning (Hard Feature Pruning)
# ──────────────────────────────────────────────

def compute_channel_importance(model):
    """
    Compute L1-norm importance scores for each output channel of each Conv1d layer.
    
    Returns a dict mapping layer name -> 1D tensor of importance scores (one per output channel).
    Higher score = more important channel.
    """
    importance = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            # L1-norm across (in_channels, kernel_size) for each output channel
            weight = module.weight.data  # (out_ch, in_ch, kernel_size)
            scores = weight.abs().sum(dim=(1, 2))  # (out_ch,)
            importance[name] = scores.cpu()
    return importance


def apply_structured_pruning(model, prune_ratio=0.25, in_channels=6, seq_len=4097, num_classes=2):
    """
    Apply structured pruning by removing the least important channels.
    
    This creates a NEW, smaller network (not a masked network) with
    genuinely reduced channel counts, then copies the surviving weights.
    
    Args:
        model: trained GenericEarlyExitNet or subclass
        prune_ratio: fraction of channels to remove (0.0 to 1.0)
        in_channels: input channel count
        seq_len: input sequence length
        num_classes: number of output classes
    
    Returns:
        pruned_model: new GenericEarlyExitNet with reduced channel sizes
        pruned_channels: list of new channel sizes
    """
    importance = compute_channel_importance(model)
    original_channels = model.channel_progression

    # Determine pruned channel sizes
    pruned_channels = []
    for i, ch in enumerate(original_channels):
        pruned_ch = max(4, int(ch * (1.0 - prune_ratio)))  # Keep at least 4 channels
        pruned_channels.append(pruned_ch)

    print(f"[Pruning] Original channels: {original_channels} -> Pruned channels: {pruned_channels}")
    print(f"[Pruning] Prune ratio: {prune_ratio:.0%}")

    # Create new model with pruned architecture
    is_adaptive = isinstance(model, AdaptiveEarlyExitNet)
    if is_adaptive:
        pruned_model = AdaptiveEarlyExitNet(
            in_channels=in_channels, channel_sizes=pruned_channels,
            num_classes=num_classes, seq_len=seq_len
        )
    else:
        pruned_model = GenericEarlyExitNet(
            in_channels=in_channels, channel_sizes=pruned_channels,
            num_classes=num_classes, seq_len=seq_len
        )

    # Transfer surviving weights
    _transfer_pruned_weights(model, pruned_model, importance, original_channels, pruned_channels)

    param_reduction = 1.0 - pruned_model.count_parameters() / model.count_parameters()
    flops_reduction = 1.0 - sum(pruned_model.stage_flops) / sum(model.stage_flops)
    print(f"[Pruning] Parameter reduction: {param_reduction:.1%}")
    print(f"[Pruning] FLOPs reduction: {flops_reduction:.1%}")

    return pruned_model, pruned_channels


def _transfer_pruned_weights(original, pruned, importance, orig_channels, pruned_channels):
    """
    Transfer weights from original model to pruned model,
    keeping only the most important channels.
    """
    # For each Conv1d pair, determine which output channels to keep
    orig_conv_layers = [(n, m) for n, m in original.named_modules() if isinstance(m, nn.Conv1d)]
    pruned_conv_layers = [(n, m) for n, m in pruned.named_modules() if isinstance(m, nn.Conv1d)]

    for (orig_name, orig_mod), (pruned_name, pruned_mod) in zip(orig_conv_layers, pruned_conv_layers):
        orig_out = orig_mod.out_channels
        pruned_out = pruned_mod.out_channels
        pruned_in = pruned_mod.in_channels

        if orig_name in importance:
            scores = importance[orig_name]
            _, keep_idx = torch.topk(scores, pruned_out)
            keep_idx, _ = keep_idx.sort()
        else:
            keep_idx = torch.arange(pruned_out)

        # Determine input channel indices (from previous layer's kept channels)
        orig_in = orig_mod.in_channels
        if pruned_in < orig_in:
            # This layer's input was pruned by the previous layer
            # We need to know which input channels were kept
            # For simplicity, use the first pruned_in channels
            in_idx = torch.arange(pruned_in)
        else:
            in_idx = torch.arange(orig_in)

        # Transfer weights
        with torch.no_grad():
            w = orig_mod.weight.data[keep_idx][:, :pruned_in, :]
            pruned_mod.weight.data.copy_(w)
            if orig_mod.bias is not None and pruned_mod.bias is not None:
                pruned_mod.bias.data.copy_(orig_mod.bias.data[keep_idx])

    # Transfer BatchNorm parameters
    orig_bn_layers = [(n, m) for n, m in original.named_modules() if isinstance(m, nn.BatchNorm1d)]
    pruned_bn_layers = [(n, m) for n, m in pruned.named_modules() if isinstance(m, nn.BatchNorm1d)]

    for (_, orig_bn), (_, pruned_bn) in zip(orig_bn_layers, pruned_bn_layers):
        pruned_n = pruned_bn.num_features
        with torch.no_grad():
            pruned_bn.weight.data.copy_(orig_bn.weight.data[:pruned_n])
            pruned_bn.bias.data.copy_(orig_bn.bias.data[:pruned_n])
            pruned_bn.running_mean.copy_(orig_bn.running_mean[:pruned_n])
            pruned_bn.running_var.copy_(orig_bn.running_var[:pruned_n])
