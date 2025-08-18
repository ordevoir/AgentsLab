from __future__ import annotations
from tensordict import TensorDictBase

def infer_obs_dim(env) -> int:
    obs_spec = env.observation_spec
    if isinstance(obs_spec, dict) or hasattr(obs_spec, "keys"):
        ospec = obs_spec.get("observation", None)
        if ospec is None:
            ospec = next(iter(obs_spec.values()))
    else:
        ospec = obs_spec
    return int(ospec.shape[-1])

def infer_action_dim(env) -> int:
    action_spec = env.action_spec
    n = getattr(getattr(action_spec, "space", None), "n", None)
    if n is not None:
        return int(n)
    return int(action_spec.shape[-1])

def td_info(td: TensorDictBase) -> str:
    lines = [f"TensorDict batch_size={td.batch_size}, device={td.device}"]
    for k in td.keys(True, True):
        try:
            v = td.get(k)
            lines.append(f"  {k}: shape={tuple(v.shape)} dtype={getattr(v, 'dtype', None)}")
        except Exception as e:
            lines.append(f"  {k}: <error {e}>")
    return "\n".join(lines)
