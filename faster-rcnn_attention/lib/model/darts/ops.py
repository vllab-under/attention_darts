""" Operations """
import torch
import torch.nn as nn
from . import genotypes as gt
import torch.nn.functional as F
# import genotypes as gt

OPS = {
    'total': lambda gate_channels, levels, reduction_ratio, kernel_size : 
    FusionAttention(gate_channels=gate_channels, levels=levels, reduction_ratio=reduction_ratio, kernel_size=kernel_size),
    'channel': lambda gate_channels, levels, reduction_ratio, kernel_size : 
    FusionAttention(gate_channels=gate_channels, levels=levels, reduction_ratio=reduction_ratio, kernel_size=kernel_size, no_spatial=True),
    'spatial': lambda gate_channels, levels, reduction_ratio, kernel_size : 
    FusionAttention(gate_channels=gate_channels, levels=levels, reduction_ratio=reduction_ratio, kernel_size=kernel_size, no_channel=True),
    'identity': lambda gate_channels, levels, reduction_ratio, kernel_size : 
    FusionAttention(gate_channels=gate_channels, levels=levels, reduction_ratio=reduction_ratio, kernel_size=kernel_size, no_channel=True, no_spatial=True)
}

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, gate_channels, levels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.gate_channels = gate_channels
        self.levels = levels
        self.mlp = nn.ModuleList()
        for _ in range(levels):
            if reduction_ratio != 1:
                self.mlp.append(nn.Sequential(
                    Flatten(),
                    nn.Linear(gate_channels, gate_channels // reduction_ratio),
                    nn.ReLU(),
                    nn.Linear(gate_channels // reduction_ratio, gate_channels // levels)
                ))
            else:
                self.mlp.append(nn.Sequential(
                    Flatten(),
                    nn.Linear(gate_channels, gate_channels // levels)
                ))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        attention = torch.cat(x, dim=1)
        scale = []
        for i in range(self.levels):
            for pool_type in self.pool_types:
                if pool_type == 'avg':
                    avg_pool = F.avg_pool2d(attention, (attention.size(2), attention.size(3)),
                                            stride=(attention.size(2), attention.size(3)))
                    channel_att_raw = self.mlp[i](avg_pool)
                elif pool_type == 'max':
                    max_pool = F.max_pool2d(attention, (attention.size(2), attention.size(3)),
                                            stride=(attention.size(2), attention.size(3)))
                    channel_att_raw = self.mlp[i](max_pool)
                elif pool_type == 'lp':
                    lp_pool = F.lp_pool2d(attention, 2, (attention.size(2), attention.size(3)),
                                          stride=(attention.size(2), attention.size(3)))
                    channel_att_raw = self.mlp[i](lp_pool)
                elif pool_type == 'lse':
                    # LSE pool only
                    lse_pool = logsumexp_2d(attention)
                    channel_att_raw = self.mlp[i](lse_pool)

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw
            scale.append(torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x[i]))
        return scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, levels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.levels = levels
        self.compress = ChannelPool()
        self.spatial = nn.ModuleList()
        for _ in range(levels):
            self.spatial.append(
                BasicConv(2 * levels, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False))

    def forward(self, x):
        attention = torch.cat([(self.compress(_)) for _ in x], dim=1)
        scale = [torch.sigmoid(self.spatial[i](attention)) for i in range(self.levels)]
        return scale


class FusionAttention(nn.Module):
    def __init__(self, gate_channels, levels, reduction_ratio=16, kernel_size=7, pool_types=['avg', 'max'],
                 no_channel=False, no_spatial=False):
        super(FusionAttention, self).__init__()
        self.levels = levels
        self.no_channel = no_channel
        self.no_spatial = no_spatial
        if not no_channel:
            self.channel_attention = ChannelAttention(gate_channels, levels, reduction_ratio, pool_types)
        if not no_spatial:
            self.spatial_attention = SpatialAttention(levels, kernel_size)

    def forward(self, x):
        if not self.no_channel:
            channel_attention_maps = self.channel_attention(x)
            x = [x[level] * channel_attention_maps[level] for level in range(self.levels)]

        if not self.no_spatial:
            spatial_attention_maps = self.spatial_attention(x)
            x = [x[level] * spatial_attention_maps[level] for level in range(self.levels)]
        return sum(x)


class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, gate_channels, levels, reduction_ratio, kernel_size):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](gate_channels, levels, reduction_ratio, kernel_size)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))