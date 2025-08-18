from __future__ import annotations
import torch.nn as nn
from torchrl.modules import MLP, MultiAgentMLP

def make_mlp(in_features: int, out_features: int, num_cells: int = 64, depth: int = 2, device=None) -> nn.Module:
    return MLP(
        in_features=in_features,
        out_features=out_features,
        depth=depth,
        num_cells=[num_cells] * depth,
        activation_class=nn.ReLU,
        device=device,
    )

def make_multiagent_mlp(
    obs_dim_per_agent: int,
    out_dim_per_agent: int,
    n_agents: int,
    depth: int = 2,
    num_cells: int = 256,
    device=None,
    centralized: bool = False,
    share_params: bool = True,
):
    return MultiAgentMLP(
        n_agent_inputs=obs_dim_per_agent,
        n_agent_outputs=out_dim_per_agent,
        n_agents=n_agents,
        centralised=centralized,
        share_params=share_params,
        depth=depth,
        num_cells=num_cells,
        activation_class=nn.Tanh,
        device=device,
    )
