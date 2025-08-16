from __future__ import annotations
from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agentslab.networks.mlp import MLP

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    hidden_sizes: tuple[int, int] = (128, 128)
    buffer_size: int = 100_000
    batch_size: int = 64
    start_eps: float = 1.0
    end_eps: float = 0.05
    eps_decay_steps: int = 50_000
    target_update_interval: int = 1_000
    tau: float = 1.0  # 1.0 => hard update, <1.0 => soft update
    grad_clip: float | None = None

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, int]) -> None:
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden_sizes=hidden_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: bool) -> None:
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: DQNConfig, device: torch.device) -> None:
        self.device = device
        self.q = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.target_q = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.start_eps = cfg.start_eps
        self.end_eps = cfg.end_eps
        self.eps_decay_steps = cfg.eps_decay_steps
        self.target_update_interval = cfg.target_update_interval
        self.tau = cfg.tau
        self.grad_clip = cfg.grad_clip
        self.total_steps = 0
        self.act_dim = act_dim

    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.eps_decay_steps))
        return self.start_eps + frac * (self.end_eps - self.start_eps)

    def select_action(self, obs: np.ndarray) -> int:
        self.total_steps += 1
        if random.random() < self.epsilon():
            return random.randrange(self.act_dim)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q(obs_t)
            return int(torch.argmax(qvals, dim=1).item())

    def update(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(-1)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Compute targets
        with torch.no_grad():
            next_q = self.target_q(s2_t).max(dim=1, keepdim=True).values
            target = r_t + (1 - d_t) * self.gamma * next_q

        q_sa = self.q(s_t).gather(1, a_t)
        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.total_steps % self.target_update_interval == 0:
            if self.tau >= 1.0:
                self.target_q.load_state_dict(self.q.state_dict())
            else:
                # soft update
                with torch.no_grad():
                    for p, tp in zip(self.q.parameters(), self.target_q.parameters()):
                        tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return {"loss/q": float(loss.item())}
