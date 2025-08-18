from dataclasses import dataclass
from typing import Sequence
import torch.nn as nn
from torchrl.modules import ValueOperator
from .networks import build_mlp, MLPConfig

@dataclass
class ValueConfig:
    obs_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "tanh"

def build_value(cfg: ValueConfig) -> ValueOperator:
    net = build_mlp(MLPConfig(in_dim=cfg.obs_dim, hidden_sizes=cfg.hidden_sizes, out_dim=1, activation=cfg.activation))
    return ValueOperator(module=net, in_keys=["observation"])
