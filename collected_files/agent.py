
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agentslab.networks.mlp import MLP
from .replay_buffer import ReplayBuffer

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    start_learning_after: int = 1_000
    target_update_every: int = 1_000
    tau: float = 1.0  # 1.0 -> hard update; <1.0 -> soft update
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    huber_delta: float = 1.0
    clip_grad_norm: float = 10.0

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.model = MLP(input_dim, output_dim, hidden_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int],
        config: DQNConfig,
        device: torch.device,
    ) -> None:
        self.device = device
        self.cfg = config
        self.q = QNetwork(obs_dim, n_actions, hidden_sizes).to(device)
        self.q_target = QNetwork(obs_dim, n_actions, hidden_sizes).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size)

        self.total_steps = 0
        self.n_actions = n_actions

    def epsilon(self) -> float:
        if self.cfg.eps_decay_steps <= 0:
            return self.cfg.eps_end
        fraction = min(1.0, self.total_steps / self.cfg.eps_decay_steps)
        return float(self.cfg.eps_start + fraction * (self.cfg.eps_end - self.cfg.eps_start))

    @torch.no_grad()
    def act(self, obs: np.ndarray, exploit: bool = False) -> int:
        eps = 0.0 if exploit else self.epsilon()
        if np.random.rand() < eps:
            return int(np.random.randint(self.n_actions))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q(obs_t)
        return int(torch.argmax(q_values, dim=-1).item())

    def push(self, *transition) -> None:
        self.buffer.push(*transition)

    def _soft_update(self) -> None:
        if self.cfg.tau >= 1.0:
            self.q_target.load_state_dict(self.q.state_dict())
        else:
            with torch.no_grad():
                for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                    p_t.mul_(1.0 - self.cfg.tau)
                    p_t.add_(self.cfg.tau * p)

    def train_step(self) -> Tuple[float, float]:
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.start_learning_after):
            return 0.0, 0.0

        states, actions, next_states, rewards, dones = self.buffer.sample(self.cfg.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Q(s,a)
        q_values = self.q(states).gather(1, actions)

        with torch.no_grad():
            # Max over next actions from target network
            next_q = self.q_target(next_states).max(1, keepdim=True).values
            targets = rewards + (1.0 - dones) * self.cfg.gamma * next_q

        td_error = targets - q_values
        # Huber loss
        loss = torch.where(
            td_error.abs() <= self.cfg.huber_delta,
            0.5 * td_error.pow(2),
            self.cfg.huber_delta * (td_error.abs() - 0.5 * self.cfg.huber_delta),
        ).mean()

        self.optim.zero_grad()
        loss.backward()
        if self.cfg.clip_grad_norm is not None and self.cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.clip_grad_norm)
        self.optim.step()

        self.total_steps += 1
        if self.total_steps % self.cfg.target_update_every == 0:
            self._soft_update()

        return float(loss.item()), float(td_error.abs().mean().item())
