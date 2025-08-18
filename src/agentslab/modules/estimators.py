from dataclasses import dataclass
from torchrl.objectives.value import GAE
from typing import Optional

@dataclass
class GAEConfig:
    gamma: float = 0.99
    lam: float = 0.95
    average_gae: bool = True  # center advantages

def build_gae(cfg: GAEConfig, value_module) -> GAE:
    return GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lam,
        value_network=value_module,
        average_gae=cfg.average_gae,
    )
