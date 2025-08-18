from dataclasses import dataclass
from typing import Optional, Sequence, Union, Dict, Any

import torch
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import Compose, ObservationNorm, DoubleToFloat, StepCounter

@dataclass
class GymEnvConfig:
    env_id: str = "InvertedDoublePendulum-v4"
    render_mode: Optional[str] = None
    norm_obs: bool = True
    init_norm_iter: int = 1000
    max_steps: Optional[int] = None  # if not None, counted by StepCounter
    device: Union[str, torch.device] = "cpu"
    seed: Optional[int] = 0

def make_gym_env(cfg: GymEnvConfig) -> TransformedEnv:
    base = GymEnv(cfg.env_id, device=cfg.device, render_mode=cfg.render_mode)
    if cfg.seed is not None:
        base.set_seed(cfg.seed)
    transforms = []
    if cfg.norm_obs:
        transforms.append(ObservationNorm(in_keys=["observation"]))
    transforms += [DoubleToFloat(), StepCounter()]
    env = TransformedEnv(base, Compose(*transforms))
    # init stats for ObservationNorm
    if cfg.norm_obs and cfg.init_norm_iter and cfg.init_norm_iter > 0:
        obs_norm = env.transform[0]  # first of Compose
        obs_norm.init_stats(num_iter=cfg.init_norm_iter, reduce_dim=0)
    return env
