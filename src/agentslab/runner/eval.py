from __future__ import annotations

import torch
from tqdm.auto import tqdm
from tensordict.nn import TensorDictSequential as Seq

from ..envs.gym import make_gym_env

def evaluate_policy(policy: Seq, env_id: str, device: torch.device, episodes: int = 10, seed: int = 0, max_steps=None):
    env = make_gym_env(env_id, seed=seed, record_video=False, max_steps=max_steps, render_mode=None)
    env.set_seed(seed)
    policy = policy.to(device).eval()
    returns = []
    pbar = tqdm(range(episodes), desc="Eval", unit="ep")
    for ep in pbar:
        td = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                td = td.to(device)
                td = policy(td)
                action = td.get("action")
            td = env.step(action)
            r = td.get("reward")
            done = (td.get("done") | td.get("terminated") | td.get("truncated")).any().item()
            ep_ret += float(r.sum().item()) if r is not None else 0.0
        returns.append(ep_ret)
        pbar.set_postfix({"return": ep_ret})
    env.close()
    return {"mean_return": float(sum(returns)/len(returns)), "returns": returns}
