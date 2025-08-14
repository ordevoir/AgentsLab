from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class EpisodeBuffers:
    log_probs: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def clear(self):
        self.log_probs.clear()
        self.rewards.clear()


class ReinforcePolicy(nn.Module):
    def __init__(self, backbone: nn.Module, device: torch.device):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.buffers = EpisodeBuffers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def act_eval(self, obs_np: np.ndarray) -> int:
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self(x)
        dist = Categorical(logits=logits)
        return int(dist.sample().item())

    def act_train(self, obs_np: np.ndarray) -> int:
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        self.buffers.log_probs.append(dist.log_prob(a).squeeze(0))
        return int(a.item())

    def compute_loss_and_rtg(self, gamma: float) -> Tuple[torch.Tensor, np.ndarray]:
        T = len(self.buffers.rewards)
        rtg = np.zeros(T, dtype=np.float32)
        g = 0.0
        for t in reversed(range(T)):
            g = float(self.buffers.rewards[t]) + gamma * g
            rtg[t] = g
        rtg_t = torch.as_tensor(rtg, dtype=torch.float32, device=self.device)
        rtg_t = (rtg_t - rtg_t.mean()) / (rtg_t.std() + 1e-8)
        logp = torch.stack(self.buffers.log_probs).to(self.device)
        loss = -(logp * rtg_t).mean()
        return loss, rtg