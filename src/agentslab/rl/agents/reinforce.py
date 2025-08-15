from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn

@dataclass
class Trajectory:
    log_probs: List[torch.Tensor]
    rewards: List[float]

class REINFORCEAgent:
    def __init__(
        self,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        normalize_returns: bool = True,
        device: str = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.normalize_returns = normalize_returns
        self.device = device

    # @torch.no_grad()
    def select_action(self, obs) -> Tuple[int, torch.Tensor]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).squeeze(0)
        return int(action.item()), log_prob

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        G = []
        g = 0.0
        for r in reversed(rewards):
            g = r + self.gamma * g
            G.append(g)
        G.reverse()
        returns = torch.tensor(G, dtype=torch.float32, device=self.device)
        if self.normalize_returns and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, trajectory: Trajectory) -> float:
        returns = self.compute_returns(trajectory.rewards)
        # Concatenate log_probs into tensor
        log_probs = torch.stack(trajectory.log_probs)
        loss = -(log_probs * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
