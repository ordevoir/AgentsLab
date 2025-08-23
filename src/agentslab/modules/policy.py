import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.data.tensor_specs import OneHotDiscreteTensorSpec, BoundedTensorSpec
from torch.distributions import OneHotCategorical

def is_acts_discrete(action_spec):
    if isinstance(action_spec, OneHotDiscreteTensorSpec):
        return True
    elif isinstance(action_spec, BoundedTensorSpec):
        return False
    else:
        raise TypeError(f"Неизвестный тип action_spec: {type(action_spec)}")

def build_stochastic_actor(network, action_spec, return_log_prob=True):
    # Дискретные (one-hot) действия
    if isinstance(action_spec, OneHotDiscreteTensorSpec):
        td_module = TensorDictModule(network, in_keys=["observation"], out_keys=["logits"])
        return ProbabilisticActor(
            module=td_module,
            in_keys=["logits"],
            distribution_class=OneHotCategorical,
            return_log_prob=return_log_prob,
            spec=action_spec,
        )
    # Непрерывные действия (ограниченные low/high)
    elif isinstance(action_spec, BoundedTensorSpec):
        net_with_extractor = nn.Sequential(network, NormalParamExtractor())
        td_module = TensorDictModule(net_with_extractor, in_keys=["observation"], out_keys=["loc", "scale"])
        return ProbabilisticActor(
            module=td_module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=return_log_prob,
            distribution_kwargs={"low": action_spec.low, "high": action_spec.high},
            spec=action_spec,
        )
    else:
        raise TypeError(f"Неизвестный тип action_spec: {type(action_spec)}")