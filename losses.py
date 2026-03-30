import torch
import torch.nn as nn

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
