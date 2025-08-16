# src/agentslab/rl/agents/dqn_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agentslab.networks.mlp import MLP


@dataclass
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 50_000

    def value(self, step: int) -> float:
        frac = min(step / max(1, self.decay_steps), 1.0)
        return float(self.start + frac * (self.end - self.start))


class ReplayBuffer:
    def __init__(self, obs_dim: int, buffer_size: int, device: torch.device) -> None:
        self.obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((buffer_size,), dtype=np.int64)
        self.rews = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.max_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(self, obs, act, rew, next_obs, done) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[idx], device=self.device)
        acts = torch.as_tensor(self.acts[idx], device=self.device)
        rews = torch.as_tensor(self.rews[idx], device=self.device)
        next_obs = torch.as_tensor(self.next_obs_buf[idx], device=self.device)
        dones = torch.as_tensor(self.dones[idx], device=self.device)
        return obs, acts, rews, next_obs, dones


class DQNAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes=(128, 128),
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.q = MLP(obs_dim, n_actions, hidden_sizes)
        self.target_q = MLP(obs_dim, n_actions, hidden_sizes)
        self.target_q.load_state_dict(self.q.state_dict())
        self.gamma = gamma
        self.device = device or torch.device("cpu")
        self.to(self.device)
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            # random action
            return int(np.random.randint(0, self.q.net[-1].out_features))
        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        q_values = self.q(obs_t)
        return int(torch.argmax(q_values, dim=1).item())

    def update(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        target_update_freq: int,
        global_step: int,
    ) -> Tuple[float, float]:
        obs, acts, rews, next_obs, dones = buffer.sample(batch_size)
        q_values = self.q(obs).gather(1, acts.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_q(next_obs).max(dim=1).values
            targets = rews + (1.0 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, targets)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optim.step()

        if global_step % target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        avg_q = float(q_values.mean().item())
        return float(loss.item()), avg_q
