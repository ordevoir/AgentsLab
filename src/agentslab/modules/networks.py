from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Sequence, Optional, List

@dataclass
class MLPConfig:
    in_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    out_dim: int = 0
    activation: str = "tanh"  # "relu", "tanh", "elu", etc.
    layer_norm: bool = False

def _act(name: str):
    name = name.lower()
    if name == "relu": return nn.ReLU()
    if name == "elu": return nn.ELU()
    if name == "gelu": return nn.GELU()
    return nn.Tanh()

def build_mlp(cfg: MLPConfig) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = cfg.in_dim
    for h in cfg.hidden_sizes:
        layers += [nn.Linear(d, h), _act(cfg.activation)]
        d = h
        if cfg.layer_norm:
            layers.append(nn.LayerNorm(d))
    if cfg.out_dim and cfg.out_dim > 0:
        layers.append(nn.Linear(d, cfg.out_dim))
    return nn.Sequential(*layers)
