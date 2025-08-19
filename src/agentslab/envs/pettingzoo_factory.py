from dataclasses import dataclass
import torch
from torchrl.envs.libs.pettingzoo import PettingZooEnv
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import RewardSum, Compose

@dataclass
class PettingZooConfig:
    domain: str = "mpe"
    task: str = "simple_v2"  # e.g., simple_v2
    parallel: bool = True
    device: torch.device | str = "cpu"
    use_action_mask: bool = False
    sum_reward: bool = True

def make_pettingzoo_env(cfg: PettingZooConfig):
    import pettingzoo
    if cfg.parallel:
        env = getattr(getattr(pettingzoo, cfg.domain), cfg.task).parallel_env()
    else:
        env = getattr(getattr(pettingzoo, cfg.domain), cfg.task).env()
    env = PettingZooEnv(env, use_mask=cfg.use_action_mask, device=cfg.device)
    transforms = []
    if cfg.sum_reward:
        transforms.append(RewardSum(in_keys=[("agents","reward")], out_keys=[("agents","episode_reward")]))
    if transforms:
        env = TransformedEnv(env, Compose(*transforms))
    return env
