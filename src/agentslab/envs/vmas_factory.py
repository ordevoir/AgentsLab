from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import torch

try:
    from torchrl.envs.libs.vmas import VmasEnv
except Exception:
    VmasEnv = None  # type: ignore

@dataclass
class VmasConfig:
    scenario: str = "spread"
    num_envs: int = 1
    frameskip: int = 1
    device: Union[str, torch.device] = "cpu"
    seed: Optional[int] = 0
    kwargs: Dict[str, Any] = None

def make_vmas_env(cfg: VmasConfig):
    if VmasEnv is None:
        raise ImportError("torchrl VMAS wrapper not available. Install vmas and torchrl extras.")
    env = VmasEnv(
        scenario=cfg.scenario,
        num_envs=cfg.num_envs,
        frameskip=cfg.frameskip,
        device=cfg.device,
        **(cfg.kwargs or {})
    )
    if cfg.seed is not None:
        env.set_seed(cfg.seed)
    return env
