from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: Sequence[int] = (256, 256),
        activation: Callable[..., nn.Module] | None = nn.ReLU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = in_features
        for h in hidden_layers:
            layers += [nn.Linear(last, h)]
            if layer_norm:
                layers += [nn.LayerNorm(h)]
            if activation is not None:
                layers += [activation()]
            last = h
        layers += [nn.Linear(last, out_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CNNAtariDQN(nn.Module):
    """Classic DQN convolutional torso for pixel observations (CHW)."""
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # compute output size lazily
        self.head = None
        self._n_actions = n_actions

    def _ensure_head(self, x: torch.Tensor):
        if self.head is None:
            with torch.no_grad():
                y = self.conv(x)
                flat = y.view(y.size(0), -1).size(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, 512),
                nn.ReLU(),
                nn.Linear(512, self._n_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_head(x)
        y = self.conv(x)
        return self.head(y)
