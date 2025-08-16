from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP builder used by both PG and DQN variants."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Iterable[int], activation: nn.Module | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.net(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
