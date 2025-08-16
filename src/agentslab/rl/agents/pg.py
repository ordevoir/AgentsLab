from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agentslab.networks.mlp import MLP


@dataclass
class PGConfig:
    gamma: float = 0.99
    lr: float = 5e-3
    grad_clip: float | None = None
    hidden_units: int = 64


class PGPolicy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_units: int) -> None:
        super().__init__()
        self.net = MLP(in_dim, out_dim, hidden=[hidden_units])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class PGAgent:
    def __init__(self, policy: PGPolicy, cfg: PGConfig, device: torch.device) -> None:
        self.device = device
        self.net = policy.to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        self.grad_clip = cfg.grad_clip
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def reset_episode(self) -> None:
        self.log_probs.clear()
        self.rewards.clear()

    def act(self, state: np.ndarray) -> int:
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        logits = self.net(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return int(action.item())

    def remember(self, reward: float) -> None:
        self.rewards.append(float(reward))

    def _returns(self) -> torch.Tensor:
        T = len(self.rewards)
        returns = torch.zeros(T, dtype=torch.float32, device=self.device)
        future = 0.0
        for t in reversed(range(T)):
            future = self.rewards[t] + self.gamma * future
            returns[t] = future
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self) -> float:
        if not self.log_probs:
            return 0.0
        R = self._returns()
        log_probs = torch.stack(self.log_probs)
        loss = -(log_probs * R).mean()
        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()
        self.reset_episode()
        return float(loss.item())

    def run_episode(self, env: gym.Env, max_steps: int = 500) -> tuple[float, float]:
        self.reset_episode()
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = self.act(obs)
            obs, rew, term, trunc, _ = env.step(action)
            self.remember(rew)
            total_reward += rew
            if term or trunc:
                break
        loss = self.update()
        return total_reward, loss
