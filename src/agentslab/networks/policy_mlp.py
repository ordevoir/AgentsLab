from typing import Optional, Tuple
import torch
import torch.nn as nn

_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}

class PolicyMLP(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, activation: str = "tanh"):
        super().__init__()
        act = _ACTIVATIONS.get(activation, nn.Tanh)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        # categorical over discrete actions
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
