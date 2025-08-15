from typing import Optional, Tuple
import gymnasium as gym

def make_env(env_id: str, render_mode: Optional[str] = None):
    env = gym.make(env_id, render_mode=render_mode)
    return env
