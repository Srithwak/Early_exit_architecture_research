import torch
import torch.nn as nn


class Conv1dBlock(nn.Module):
    """Standard conv block for long sequences (e.g. Bonn EEG, 4097). Downsamples /16."""
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


class Conv1dBlockSmall(nn.Module):
    """Lighter conv block for short sequences (e.g. MIT-BIH, 187). Downsamples /4."""
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


class GenericEarlyExitNet(nn.Module):
    def __init__(self, in_channels=1, channel_sizes=[32, 32, 32], num_classes=2, seq_len=4097):
        super().__init__()
        self.num_stages = len(channel_sizes)
        self.channel_progression = channel_sizes
        self.stage_flops = []

        self.entry = Conv1dBlock(in_channels, channel_sizes[0])
        self.stages = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.policies = nn.ModuleList()

        curr_seq = seq_len
        for i in range(self.num_stages):
            if i == 0:
                self.stages.append(nn.Identity())
                in_ch, out_ch = in_channels, channel_sizes[0]
            else:
                self.stages.append(Conv1dBlock(channel_sizes[i-1], channel_sizes[i]))
                in_ch, out_ch = channel_sizes[i-1], channel_sizes[i]

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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdaptiveEarlyExitNet(GenericEarlyExitNet):
    def __init__(self, in_channels=1, channel_sizes=[64, 64, 64], num_classes=2, seq_len=4097):
        super().__init__(in_channels, channel_sizes, num_classes, seq_len)
        self.gates = nn.ModuleList([ChannelGate(ch) for ch in channel_sizes])

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


class GenericEarlyExitNetSmall(nn.Module):
    """Early-exit network for short sequences (MIT-BIH / ECG, seq_len=187)."""
    def __init__(self, in_channels=6, channel_sizes=[64, 64, 64], num_classes=5, seq_len=187):
        super().__init__()
        self.num_stages = len(channel_sizes)
        self.channel_progression = channel_sizes
        self.stage_flops = []

        self.entry = Conv1dBlockSmall(in_channels, channel_sizes[0])
        self.stages = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.policies = nn.ModuleList()

        curr_seq = seq_len
        for i in range(self.num_stages):
            if i == 0:
                self.stages.append(nn.Identity())
                in_ch, out_ch = in_channels, channel_sizes[0]
            else:
                self.stages.append(Conv1dBlockSmall(channel_sizes[i-1], channel_sizes[i]))
                in_ch, out_ch = channel_sizes[i-1], channel_sizes[i]

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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdaptiveEarlyExitNetSmall(GenericEarlyExitNetSmall):
    def __init__(self, in_channels=6, channel_sizes=[64, 64, 64], num_classes=5, seq_len=187):
        super().__init__(in_channels, channel_sizes, num_classes, seq_len)
        self.gates = nn.ModuleList([ChannelGate(ch) for ch in channel_sizes])

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


# --- Loss functions ---

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
            return task_losses[-1].mean()

        if self.energy_lambda == 0.0:
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


# --- Structured pruning ---

def compute_channel_importance(model):
    importance = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            weight = module.weight.data
            scores = weight.abs().sum(dim=(1, 2))
            importance[name] = scores.cpu()
    return importance


def _transfer_pruned_weights(original, pruned, importance):
    orig_convs = [(n, m) for n, m in original.named_modules() if isinstance(m, nn.Conv1d)]
    pruned_convs = [(n, m) for n, m in pruned.named_modules() if isinstance(m, nn.Conv1d)]

    for (orig_name, orig_mod), (_, pruned_mod) in zip(orig_convs, pruned_convs):
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

    orig_bns = [(n, m) for n, m in original.named_modules() if isinstance(m, nn.BatchNorm1d)]
    pruned_bns = [(n, m) for n, m in pruned.named_modules() if isinstance(m, nn.BatchNorm1d)]

    for (_, ob), (_, pb) in zip(orig_bns, pruned_bns):
        n = pb.num_features
        with torch.no_grad():
            pb.weight.data.copy_(ob.weight.data[:n])
            pb.bias.data.copy_(ob.bias.data[:n])
            pb.running_mean.copy_(ob.running_mean[:n])
            pb.running_var.copy_(ob.running_var[:n])


def apply_structured_pruning(model, prune_ratio=0.25, in_channels=6, seq_len=4097, num_classes=2):
    importance = compute_channel_importance(model)
    original_channels = model.channel_progression
    pruned_channels = [max(4, int(ch * (1.0 - prune_ratio))) for ch in original_channels]

    print(f"[Pruning] {original_channels} -> {pruned_channels} (ratio={prune_ratio:.0%})")

    # pick the right class
    if isinstance(model, AdaptiveEarlyExitNet):
        pruned_model = AdaptiveEarlyExitNet(in_channels, pruned_channels, num_classes, seq_len)
    elif isinstance(model, AdaptiveEarlyExitNetSmall):
        pruned_model = AdaptiveEarlyExitNetSmall(in_channels, pruned_channels, num_classes, seq_len)
    elif isinstance(model, GenericEarlyExitNetSmall):
        pruned_model = GenericEarlyExitNetSmall(in_channels, pruned_channels, num_classes, seq_len)
    else:
        pruned_model = GenericEarlyExitNet(in_channels, pruned_channels, num_classes, seq_len)

    _transfer_pruned_weights(model, pruned_model, importance)

    param_red = 1.0 - pruned_model.count_parameters() / model.count_parameters()
    flops_red = 1.0 - sum(pruned_model.stage_flops) / sum(model.stage_flops)
    print(f"[Pruning] Params: -{param_red:.1%}, FLOPs: -{flops_red:.1%}")

    return pruned_model, pruned_channels
