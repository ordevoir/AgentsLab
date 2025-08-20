import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal
from tensordict.nn.distributions import NormalParamExtractor

def build_stochastic_actor(network, 
                           action_spec,
                           return_log_prob=True):
    
    net_with_extractor = nn.Sequential(network, NormalParamExtractor())
    td_module = TensorDictModule(
        net_with_extractor,
        in_keys=["observation"],
        out_keys=["loc", "scale"]
    )

    actor = ProbabilisticActor(
        module=td_module,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        return_log_prob=return_log_prob,
        distribution_kwargs={"low": action_spec.low, "high": action_spec.high},
        spec=action_spec,
    )
    return actor

