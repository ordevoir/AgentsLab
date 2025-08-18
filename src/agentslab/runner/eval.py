from __future__ import annotations
import torch

def run_episode(env, policy, device=None, render: bool = False, max_steps: int | None = None):
    td = env.reset()
    step = 0
    with torch.no_grad():
        while True:
            td = td.to(device) if device is not None else td
            td = policy(td)
            td = env.step(td)
            if render:
                try:
                    env.render()
                except Exception:
                    pass
            step += 1
            if td.get(("next", "done")).all():
                break
            if max_steps is not None and step >= max_steps:
                break
            td = td.get("next")
    return td
