
from __future__ import annotations
from typing import Tuple, Sequence, Optional
import torch

from agentslab.rl.agents.dqn.agent import DQNAgent, DQNConfig
from agentslab.rl.agents.reinforce.agent import ReinforceAgent, ReinforceConfig

def detect_algo(checkpoint: dict) -> Optional[str]:
    if "q" in checkpoint and "q_target" in checkpoint:
        return "dqn"
    if "policy" in checkpoint:
        return "reinforce"
    return None

def build_agent_from_cfg(
    algo: str,
    obs_dim: int,
    n_actions: int,
    hidden_sizes: Sequence[int],
    cfg_agent,
    device: torch.device,
):
    if algo == "dqn":
        if not isinstance(cfg_agent, DQNConfig):
            cfg_agent = DQNConfig(**cfg_agent)
        agent = DQNAgent(obs_dim, n_actions, hidden_sizes, cfg_agent, device)
        return agent
    elif algo == "reinforce":
        if not isinstance(cfg_agent, ReinforceConfig):
            cfg_agent = ReinforceConfig(**cfg_agent)
        agent = ReinforceAgent(obs_dim, n_actions, hidden_sizes, cfg_agent, device)
        return agent
    else:
        raise ValueError(f"Unknown algo: {algo}")

def load_weights(agent, checkpoint: dict, algo: str) -> None:
    if algo == "dqn":
        agent.q.load_state_dict(checkpoint["q"])
        agent.q_target.load_state_dict(checkpoint["q_target"])
        if "optim" in checkpoint:
            # Optional: don't restore optimizer for eval-only
            pass
    elif algo == "reinforce":
        agent.policy.load_state_dict(checkpoint["policy"])
        if "optim" in checkpoint:
            pass
    else:
        raise ValueError(f"Unknown algo: {algo}")
