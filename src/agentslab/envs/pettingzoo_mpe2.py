from __future__ import annotations
from torchrl.envs.libs.pettingzoo import PettingZooEnv

def wrap_pettingzoo(env, **kwargs):
    return PettingZooEnv(env, **kwargs)
