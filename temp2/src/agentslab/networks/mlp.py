from __future__ import annotations
from typing import Sequence, List
import torch
import torch.nn as nn

def _layer(in_dim: int, out_dim: int, activation: nn.Module | None) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)

class MLP(nn.Module):
    """Simple MLP with configurable hidden layers."""
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int] = (128, 128), 
                 activation: nn.Module | None = nn.ReLU()) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_sizes)
        modules: List[nn.Module] = []
        for i in range(len(dims) - 1):
            modules.append(_layer(dims[i], dims[i+1], activation))
        modules.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
