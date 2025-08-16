# src/agentslab/rl/agents/reinforce_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

from agentslab.networks.mlp import MLP


@dataclass
class ReinforceConfig:
    gamma: float = 0.99
    lr: float = 1e-2
    hidden_sizes: Iterable[int] = (128, 128)
    normalize_returns: bool = True


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(128, 128)) -> None:
        super().__init__()
        self.net = MLP(obs_dim, n_actions, hidden_sizes)

    def forward(self, obs_t: torch.Tensor) -> D.Categorical:
        logits = self.net(obs_t)
        return D.Categorical(logits=logits)

    def act(self, obs: np.ndarray) -> Tuple[int, torch.Tensor]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        pi = self.forward(obs_t)
        action = pi.sample()
        logp = pi.log_prob(action)
        return int(action.item()), logp.squeeze(0)


class ReinforceAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: ReinforceConfig) -> None:
        self.policy = CategoricalPolicy(obs_dim, n_actions, tuple(cfg.hidden_sizes))
        self.optim = optim.Adam(self.policy.parameters(), lr=float(cfg.lr))
        self.gamma = float(cfg.gamma)
        self.normalize_returns = bool(cfg.normalize_returns)

    def update(self, logps: List[torch.Tensor], rewards: List[float]) -> float:
        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        returns_t = torch.as_tensor(returns, dtype=torch.float32)

        if self.normalize_returns:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = -(torch.stack(logps) * returns_t).sum()

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optim.step()
        return float(loss.item())
