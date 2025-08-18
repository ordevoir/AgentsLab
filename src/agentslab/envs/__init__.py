from __future__ import annotations
import gymnasium as gym
from torchrl.envs import GymEnv, GymWrapper

def make_env(env_id: str, seed: int = 0, render_mode: str | None = None):
    """
    Создаёт совместимое с TorchRL окружение. Предпочтительно GymEnv.
    """
    try:
        env = GymEnv(env_id, seed=seed, render_mode=render_mode)
    except Exception:
        base = gym.make(env_id, render_mode=render_mode)
        base.reset(seed=seed)
        env = GymWrapper(base)
    return env
