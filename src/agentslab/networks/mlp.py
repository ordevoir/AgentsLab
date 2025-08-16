
from typing import Iterable, List
import torch
import torch.nn as nn

class MLP(nn.Module):
    """Simple MLP for small discrete-control tasks."""

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
