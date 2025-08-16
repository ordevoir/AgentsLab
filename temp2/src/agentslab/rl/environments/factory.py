from __future__ import annotations
from typing import Tuple
import gymnasium as gym

def make_env(env_id: str, seed: int | None = None, render_mode: str | None = None):
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env
