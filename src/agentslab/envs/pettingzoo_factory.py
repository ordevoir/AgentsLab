from dataclasses import dataclass
from typing import Optional, Union
import torch

try:
    from torchrl.envs.libs.pettingzoo import PettingZooEnv
except Exception as e:
    PettingZooEnv = None  # type: ignore

@dataclass
class PettingZooConfig:
    env_maker: callable  # lambda returning a PettingZoo env instance
    parallel: bool = True
    use_mask: bool = True
    device: Union[str, torch.device] = "cpu"
    seed: Optional[int] = 0

def make_pettingzoo_env(cfg: PettingZooConfig):
    if PettingZooEnv is None:
        raise ImportError("torchrl PettingZooEnv wrapper not available. Install pettingzoo and torchrl extras.")
    env = PettingZooEnv(cfg.env_maker, parallel=cfg.parallel, use_mask=cfg.use_mask, device=cfg.device)
    if cfg.seed is not None:
        env.set_seed(cfg.seed)
    return env
