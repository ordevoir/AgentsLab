from __future__ import annotations
from torchrl.envs.libs.gym import GymEnv

def make_gym_env(env_id: str = "CartPole-v1", device=None, **kwargs):
    return GymEnv(env_id, device=device, **kwargs)
