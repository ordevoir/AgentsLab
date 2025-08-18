from __future__ import annotations
from typing import Any
from tensordict import TensorDictBase
from torchrl.envs import EnvBase

def describe_env(env: EnvBase) -> str:
    s = []
    s.append(f"Env: {getattr(env, 'name', type(env).__name__)}")
    try:
        s.append(f"Observation spec: {env.observation_spec}")
    except Exception:
        pass
    try:
        s.append(f"Action spec: {env.action_spec}")
    except Exception:
        pass
    try:
        s.append(f"Reward spec: {env.reward_spec}")
    except Exception:
        pass
    return "\n".join(s)

def safe_td_get(td: TensorDictBase, key: str, default: Any=None):
    try:
        return td.get(key)
    except Exception:
        return default
