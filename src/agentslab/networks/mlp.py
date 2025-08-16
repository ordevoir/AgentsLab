# src/agentslab/networks/mlp.py
from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP with configurable hidden sizes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: Iterable[int] = (128, 128),
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, int(h)), activation()]
            last = int(h)
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
