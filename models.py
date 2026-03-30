import torch
import torch.nn as nn

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
