from dataclasses import dataclass
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from typing import Optional

@dataclass
class SchedulerConfig:
    name: str = "cosine"  # "cosine", "exp", "none"
    T_max: Optional[int] = None  # for cosine: number of outer iters
    gamma: float = 0.999  # for exp

def build_scheduler(optimizer, cfg: SchedulerConfig):
    if cfg.name == "cosine":
        if cfg.T_max is None:
            raise ValueError("SchedulerConfig.T_max must be set for cosine annealing.")
        return CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=0.0)
    if cfg.name == "exp":
        return ExponentialLR(optimizer, gamma=cfg.gamma)
    return None
