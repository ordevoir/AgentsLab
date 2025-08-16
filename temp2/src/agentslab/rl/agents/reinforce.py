from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from agentslab.networks.mlp import MLP

@dataclass
class ReinforceConfig:
    gamma: float = 0.99
    lr: float = 1e-2
    hidden_sizes: tuple[int, int] = (128, 128)
    grad_clip: float | None = None

class ReinforceAgent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: ReinforceConfig) -> None:
        super().__init__()
        self.policy = MLP(obs_dim, act_dim, hidden_sizes=cfg.hidden_sizes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        self.grad_clip = cfg.grad_clip

    def select_action(self, obs: torch.Tensor) -> Tuple[int, torch.Tensor]:
        logits = self.policy(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.item(), logp

    def update(self, logps: List[torch.Tensor], rewards: List[float]) -> float:
        returns: List[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = 0.0
        for logp, Gt in zip(logps, returns_t):
            loss -= logp * Gt

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.item())
