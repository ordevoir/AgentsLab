from __future__ import annotations
from typing import List
import numpy as np


def evaluate(env_maker, policy, episodes: int = 5, render: bool = False) -> List[float]:
    returns: List[float] = []
    env = env_maker()
    for _ in range(episodes):
        obs, info = env.reset()
        ep_ret = 0.0
        policy.eval()
        while True:
            action = policy.act_eval(obs)
            obs, rew, terminated, truncated, info = env.step(action)
            ep_ret += float(rew)
            if terminated or truncated:
                break
        returns.append(ep_ret)
    env.close()
    return returns

