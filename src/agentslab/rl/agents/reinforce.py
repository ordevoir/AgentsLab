
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from agentslab.networks.mlp import MLP

@dataclass
class ReinforceConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    entropy_coef: float = 0.0
    normalize_returns: bool = True
    clip_grad_norm: float = 10.0

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.body = MLP(input_dim, n_actions, hidden_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

class ReinforceAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int],
        config: ReinforceConfig,
        device: torch.device,
    ) -> None:
        self.device = device
        self.cfg = config
        self.policy = PolicyNetwork(obs_dim, n_actions, hidden_sizes).to(device)
        self.optim = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)

    def act(self, obs: np.ndarray) -> Tuple[int, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def update(self, log_probs: List[torch.Tensor], rewards: List[float]) -> float:
        # Compute discounted returns
        G = []
        g = 0.0
        for r in reversed(rewards):
            g = r + self.cfg.gamma * g
            G.append(g)
        G = list(reversed(G))
        returns = torch.as_tensor(G, dtype=torch.float32, device=self.device)
        if self.cfg.normalize_returns and returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        log_probs_t = torch.stack(log_probs)
        loss = -(log_probs_t * returns).mean()

        # Optional entropy bonus (encourages exploration)
        # We reconstruct the distribution for entropy using stored logits where possible.
        # If you want exact entropy, store logits each step.
        # Here we approximate by re-computing from the actions' log_probs magnitude.
        # In practice, keep logits if entropy_coef > 0.0.
        if self.cfg.entropy_coef > 0.0:
            # Small heuristic: encourage larger |log_prob| -> more entropy
            entropy = -log_probs_t.mean()
            loss = loss - self.cfg.entropy_coef * entropy

        self.optim.zero_grad()
        loss.backward()
        if self.cfg.clip_grad_norm is not None and self.cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.clip_grad_norm)
        self.optim.step()

        return float(loss.item())
