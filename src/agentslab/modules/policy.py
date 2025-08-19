from dataclasses import dataclass
from typing import Sequence, Optional, Union
import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal
from tensordict.nn.distributions import NormalParamExtractor

from agentslab.modules.networks import build_mlp, MLPConfig

@dataclass
class StochasticPolicyConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "tanh"
    min_std: float = 1e-4  # handled by NormalParamExtractor

def build_stochastic_actor(cfg: StochasticPolicyConfig, action_spec) -> ProbabilisticActor:
    # base MLP to 2*action_dim, then param extractor
    net = nn.Sequential(
        build_mlp(MLPConfig(in_dim=cfg.obs_dim, hidden_sizes=cfg.hidden_sizes, out_dim=2*cfg.action_dim, activation=cfg.activation)),
        NormalParamExtractor(),
    )
    td_module = TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
    actor = ProbabilisticActor(
        module=td_module,
        distribution_class=TanhNormal,
        distribution_kwargs={"low": action_spec.low, "high": action_spec.high},
        spec=action_spec,
        return_log_prob=True,
        in_keys=["loc", "scale"],
    )
    return actor
