from dataclasses import dataclass
from typing import Union, Optional
import torch
from tqdm import tqdm
from torchrl.envs.utils import ExplorationType, set_exploration_type

@dataclass
class EvalConfig:
    steps: int = 1000
    device: Union[str, torch.device] = "cpu"

@torch.no_grad()
def evaluate_policy(env, policy, cfg: EvalConfig):
    total_reward = 0.0
    total_steps = 0
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.reset()
        pbar = tqdm(range(cfg.steps), desc="Eval", leave=False)
        for _ in pbar:
            td = policy(td)
            td = env.step(td)
            r = td.get(("next","reward")).mean().item()
            total_reward += r
            total_steps += 1
            td = td.get("next")
    return {"eval_return": total_reward, "eval_steps": total_steps}
