import torch
import torch.nn as nn
import numpy as np


class ResidualSubBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, negative_slope=0.01):
        super(ResidualSubBlock, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.residual_branch = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.final_activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        main_out = self.main_branch(x)
        residual = self.residual_branch(x)
        out = main_out + residual
        # out = self.bn(out)
        # out = self.final_activation(out)
        return out


class ConvExpert(nn.Module):
    def __init__(self, input_channels, hidden_size=32, dropout_rate=0.1):
        super(ConvExpert, self).__init__()
        self.input_norm = nn.BatchNorm2d(input_channels)
        self.sub_block1 = ResidualSubBlock(input_channels, hidden_size)
        self.sub_block2 = ResidualSubBlock(input_channels, hidden_size)
        self.sub_block3 = ResidualSubBlock(input_channels, hidden_size)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.final_conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )


        for m in self.final_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.sub_block1(x)
        x = self.sub_block2(x)
        x = self.sub_block3(x)
        x = self.dropout(x)
        out = self.final_conv(x)
        return out


class RegionRouter(nn.Module):

    def __init__(self, input_channels=103, num_regions=3, experts_per_region=3, top_k=2):
        super(RegionRouter, self).__init__()
        self.num_regions = num_regions
        self.experts_per_region = experts_per_region
        self.top_k = top_k

        self.region_gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(input_channels, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, experts_per_region),
            ) for _ in range(num_regions)
        ])

        for gate in self.region_gates:
            for m in gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):

        batch_size = x.shape[0]
        gate_logits = []

        for region_idx in range(self.num_regions):
            logit = self.region_gates[region_idx](x)  # (B, experts_per_region)
            gate_logits.append(logit.unsqueeze(1))  # (B, 1, experts_per_region)

        gate_logits = torch.cat(gate_logits, dim=1)  # (B, num_regions, experts_per_region)


        top_k_gates, top_k_indices = gate_logits.topk(
            self.top_k, dim=2, sorted=False
        )  # (B, num_regions, top_k)


        top_k_scores = torch.softmax(top_k_gates, dim=2)

        return top_k_indices, top_k_scores, gate_logits


class HyperspectralRegionMoE(nn.Module):

    def __init__(self, partition_indices, in_channels=64,
                 hidden_size=32, num_regions=3, experts_per_region=3, top_k=2, projection_dim=64):
        super(HyperspectralRegionMoE, self).__init__()

        self.partition = partition_indices
        if len(partition_indices) != 4:
            raise ValueError("partition_indices must have 4 elements（[0, c1, c1+c2, C]）")

        self.c1 = self.partition[1] - self.partition[0]
        self.c2 = self.partition[2] - self.partition[1]
        self.c3 = self.partition[3] - self.partition[2]
        assert self.c1 + self.c2 + self.c3 == in_channels

        self.num_regions = num_regions
        self.experts_per_region = experts_per_region
        self.top_k = top_k
        self.projection_dim = projection_dim

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.c1 + hidden_size, projection_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(projection_dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(self.c2 + hidden_size, projection_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(projection_dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(self.c3 + hidden_size, projection_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(projection_dim),
                nn.ReLU(inplace=True)
            )
        ])

        for proj in self.projections:
            for m in proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


        self.region_expert_pools = nn.ModuleList()
        for region_idx in range(num_regions):
            region_experts = nn.ModuleList()
            for expert_idx in range(experts_per_region):
                expert = ConvExpert(projection_dim, hidden_size)
                region_experts.append(expert)
            self.region_expert_pools.append(region_experts)


        self.router = RegionRouter(
            input_channels=in_channels,
            num_regions=num_regions,
            experts_per_region=experts_per_region,
            top_k=top_k
        )


        self.fusion = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        for m in self.fusion.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def split_hyperspectral(self, x):
        p0, p1, p2, p3 = self.partition
        sub1 = x[:, p0:p1, :, :]
        sub2 = x[:, p1:p2, :, :]
        sub3 = x[:, p2:p3, :, :]
        return [sub1, sub2, sub3]

    def cv_squared(self, x):

        eps = 1e-10
        if x.shape[0] == 0:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x , y, loss_coef=1e-2):

        batch_size, channels, height, width = x.shape

        region_inputs = self.split_hyperspectral(x)  # [(B,c1,H,W), (B,c2,H,W), (B,c3,H,W)]


        projected_inputs = []
        for region_idx in range(self.num_regions):

            projected = self.projections[region_idx](torch.cat([region_inputs[region_idx] , y] , dim=1))
            projected_inputs.append(projected)  # [(B, D, H, W) × 3]，D=projection_dim


        expert_indices, expert_scores, gate_logits = self.router(x)
        # expert_indices: (B, 3, top_k), expert_scores: (B, 3, top_k)


        load_loss = 0.0
        for region_idx in range(self.num_regions):

            region_gates = gate_logits[:, region_idx, :]

            expert_probs = torch.softmax(region_gates, dim=1)
            expert_load = expert_probs.sum(dim=0)  # (experts_per_region,)

            load_loss += self.cv_squared(expert_load)

        aux_loss = load_loss * loss_coef


        region_outputs = []
        for region_idx in range(self.num_regions):
            region_inp = projected_inputs[region_idx]

            idx = expert_indices[:, region_idx, :]  # (B, top_k)
            scores = expert_scores[:, region_idx, :]  # (B, top_k)


            region_out = torch.zeros_like(region_inp)

            for k in range(self.top_k):
                k_idx = idx[:, k]  # (B,)
                k_scores = scores[:, k].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B,1,1,1)

                for expert_idx in range(self.experts_per_region):
                    mask = (k_idx == expert_idx)
                    if not mask.any():
                        continue

                    expert_inp = region_inp[mask]
                    expert = self.region_expert_pools[region_idx][expert_idx]
                    expert_out = expert(expert_inp)
                    region_out[mask] += expert_out * k_scores[mask]

            region_outputs.append(region_out)



        combined = torch.cat(region_outputs, dim=1)

        output = self.fusion(combined)

        return output, aux_loss



