from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Sequence, List, Union
from torchrl.modules import MultiAgentMLP

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


# =========================
# 1) Конфигурация MLP
# =========================

@dataclass
class MultiAgentMLPConfig:

    n_agent_inputs: int
    n_agent_outputs: int
    n_agents: int

    # MARL-параметры:
    centralized: bool = False          # централизованная критика/политика (MAPPO) или нет (IPPO)
    share_params: bool = True          # общий набор весов между агентами

    # Система:
    device: Union[str, torch.device] = "cpu"
    
    # Архитектура:
    depth: int = 2
    num_cells: Union[int, Sequence[int]] = 256
    activation_class: type[nn.Module] = nn.Tanh
    use_td_params: bool = True


# ============================================
# 2) Функция сборки MultiAgentMLP по конфигу
# ============================================

def build_multi_agent_mlp(cfg: MultiAgentMLPConfig,
        ) -> MultiAgentMLP:
    base_network = MultiAgentMLP(
        n_agent_inputs=cfg.n_agent_inputs,
        n_agent_outputs=cfg.n_agent_outputs,
        n_agents=cfg.n_agents,
        centralized=cfg.centralized,
        share_params=cfg.share_params,
        device=cfg.device,
        depth=cfg.depth,
        num_cells=cfg.num_cells,
        activation_class=cfg.activation_class,
    )
    return base_network

