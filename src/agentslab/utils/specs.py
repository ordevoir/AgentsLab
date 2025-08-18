from __future__ import annotations
from tensordict import TensorDictBase

def pretty_specs(env) -> str:
    parts = []
    parts.append(f"Batch size: {getattr(env, 'batch_size', None)}")
    try:
        parts.append(f"Observation spec: {env.observation_spec}")
    except Exception as e:
        parts.append(f"Observation spec: <error {e}>")
    try:
        parts.append(f"Action spec: {env.action_spec}")
    except Exception as e:
        parts.append(f"Action spec: <error {e}>")
    return "\n".join(parts)

def td_summary(td: TensorDictBase) -> str:
    return str(td)
