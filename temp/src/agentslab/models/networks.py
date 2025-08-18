from __future__ import annotations
import torch.nn as nn
from torchrl.modules import MLP

def make_mlp(in_features: int, out_features: int, hidden_sizes: list[int]) -> nn.Module:
    depth = len(hidden_sizes)
    return MLP(
        in_features=in_features,
        out_features=out_features,
        depth=depth,
        num_cells=hidden_sizes if hidden_sizes else None,
        activation_class=nn.ReLU,
    )
