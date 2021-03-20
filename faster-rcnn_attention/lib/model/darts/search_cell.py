import torch
import torch.nn as nn
from . import ops
# from models import ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, gate_channels, levels, reduction_ratio, kernel_size):
        super().__init__()
        self.op = ops.MixedOp(gate_channels=gate_channels, levels=levels, reduction_ratio=reduction_ratio, kernel_size=kernel_size)

    def forward(self, x, weights):
        return self.op(x, weights)