
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from agentslab.utils.paths import logs_dir, ckpts_dir, results_dir
from agentslab.utils.seed import set_seed

@dataclass
class EnvSpec:
    id: str
    max_steps: int

def make_env(env_id: str, seed: int) -> Tuple[gym.Env, int, int]:
    env = gym.make(env_id)
    # Handle Gymnasium reset API
    obs, _ = env.reset(seed=seed)
    obs_dim = int(np.array(obs).shape[-1])
    if hasattr(env.action_space, "n"):
        n_actions = int(env.action_space.n)
    else:
        raise ValueError("This scaffold currently supports only discrete action spaces.")
    return env, obs_dim, n_actions
