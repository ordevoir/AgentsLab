from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, tensor

from agentslab.networks.mlp import MLP
from agentslab.rl.memory.replay_buffer import ReplayMemory, Transition
from agentslab.utils.schedules import epsilon_by_frame


@dataclass
class DQNConfig:
    gamma: float = 0.99
    tau: float = 0.005  # soft target update
    batch_size: int = 64
    buffer_capacity: int = 10000
    clip_grad_value: float = 5.0
    lr: float = 5e-4
    weight_decay: float = 1e-5
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay_steps: int = 2000
    hidden1: int = 128
    hidden2: int = 64


class QNet(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, hidden1: int, hidden2: int) -> None:
        super().__init__()
        self.net = MLP(in_dim, act_dim, hidden=[hidden1, hidden2])

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: DQNConfig, device: torch.device) -> None:
        self.device = device
        self.online = QNet(obs_dim, act_dim, cfg.hidden1, cfg.hidden2).to(device)
        self.target = QNet(obs_dim, act_dim, cfg.hidden1, cfg.hidden2).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.memory = ReplayMemory(cfg.buffer_capacity)
        self.optimizer = optim.Adam(
            self.online.parameters(), lr=cfg.lr, amsgrad=True, weight_decay=cfg.weight_decay
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.batch_size = cfg.batch_size
        self.clip_grad_value = cfg.clip_grad_value

        self.eps_start = cfg.eps_start
        self.eps_end = cfg.eps_end
        self.eps_decay_steps = cfg.eps_decay_steps
        self.steps = 0

    def select_action(self, state: Tensor, env: gym.Env) -> Tensor:
        eps = epsilon_by_frame(self.steps, self.eps_start, self.eps_end, self.eps_decay_steps)
        self.steps += 1
        if torch.rand(1).item() > eps:
            with torch.no_grad():
                q_values = self.online(state)
            action = q_values.max(dim=1).indices
            return action.view(1, 1)
        else:
            a = env.action_space.sample()
            return tensor([[a]], device=self.device, dtype=torch.long)

    @torch.no_grad()
    def _double_dqn_next_v(self, non_terminal_next_states: Tensor, mask: Tensor) -> Tensor:
        # Action selection by online net, evaluation by target net
        next_q_online = self.online(non_terminal_next_states)
        next_q_target = self.target(non_terminal_next_states)
        best_next_actions = next_q_online.max(1).indices.unsqueeze(1)
        next_v = torch.zeros((self.batch_size, 1), device=self.device)
        next_v[mask] = next_q_target.gather(1, best_next_actions)
        return next_v

    def optimize(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        mask = tuple(map(lambda s: s is not None, batch.next_state))
        mask_t = tensor(mask, device=self.device, dtype=torch.bool)
        non_term_next_states = [s for s in batch.next_state if s is not None]
        non_term_next_states = torch.cat(non_term_next_states) if non_term_next_states else None

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        if non_term_next_states is not None:
            next_v = self._double_dqn_next_v(non_term_next_states, mask_t)
        else:
            next_v = torch.zeros((self.batch_size, 1), device=self.device)

        q_targets = reward_batch.unsqueeze(1) + self.gamma * next_v

        q = self.online(state_batch)
        q_selected = q.gather(dim=1, index=action_batch)
        loss = self.loss_fn(q_selected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.online.parameters(), self.clip_grad_value)
        self.optimizer.step()

        # Soft target update
        with torch.no_grad():
            for t_param, o_param in zip(self.target.parameters(), self.online.parameters()):
                t_param.data.copy_(self.tau * o_param.data + (1.0 - self.tau) * t_param.data)

        return float(loss.item())
