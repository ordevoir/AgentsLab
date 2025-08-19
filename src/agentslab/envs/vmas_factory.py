from dataclasses import dataclass
import torch
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import RewardSum, Compose

@dataclass
class VMASConfig:
    scenario: str = "navigation"
    n_agents: int = 3
    continuous_actions: bool = True
    max_steps: int = 100
    num_envs: int = 64
    device: torch.device | str = "cpu"
    sum_reward: bool = True

def make_vmas_env(cfg: VMASConfig):
    env = VmasEnv(
        scenario=cfg.scenario,
        num_envs=cfg.num_envs,
        continuous_actions=cfg.continuous_actions,
        max_steps=cfg.max_steps,
        device=cfg.device,
        n_agents=cfg.n_agents,
    )
    transforms = []
    if cfg.sum_reward:
        transforms.append(RewardSum(in_keys=[("agents","reward")], out_keys=[("agents","episode_reward")]))
    if transforms:
        env = TransformedEnv(env, Compose(*transforms))
    return env
