import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import OneHotCategorical, Categorical
from tensordict.nn import InteractionType
from torchrl.data.tensor_specs import (
    OneHotDiscreteTensorSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    BoundedTensorSpec,
)


# def is_acts_discrete(action_spec):
#     if isinstance(action_spec, OneHotDiscreteTensorSpec):
#         return True
#     elif isinstance(action_spec, BoundedTensorSpec):
#         return False
#     else:
#         raise TypeError(f"Неизвестный тип action_spec: {type(action_spec)}")

def is_acts_discrete(action_spec) -> bool:
    """True для всех дискретных вариантов сред:
    OneHotDiscreteTensorSpec, DiscreteTensorSpec, MultiDiscreteTensorSpec"""
    return isinstance(
        action_spec,
        (OneHotDiscreteTensorSpec, DiscreteTensorSpec, MultiDiscreteTensorSpec),
    )


def get_num_action_logits(action_spec) -> int:
    """
    Возвращает число логитов, которое должна выпускать политика до преобразования
    в распределение (удобно для конфигурации выходного слоя MLP).

    - OneHotDiscrete / Discrete: n
    - MultiDiscrete: суммарное число категорий (на каждый дискрит отдельные логиты)
    - Bounded (Box): размерность действия (для непрерывного актора вы дальше умножаете на 2)
    """
    if isinstance(action_spec, OneHotDiscreteTensorSpec):
        # у OneHotDiscreteTensorSpec n == shape[-1]
        return int(action_spec.n)
    if isinstance(action_spec, DiscreteTensorSpec):
        return int(action_spec.n)
    if isinstance(action_spec, MultiDiscreteTensorSpec):
        # суммарные логиты под независимые Categorical по осям
        return int(action_spec.nvec.sum())
    if isinstance(action_spec, BoundedTensorSpec):
        return int(action_spec.shape[-1])
    raise TypeError(f"Неизвестный тип action_spec: {type(action_spec)}")


def build_stochastic_actor(network, action_spec, return_log_prob: bool = True):
    """
    Строит стохастического актёра под тип action_spec.

    Поддержано:
      - OneHotDiscreteTensorSpec  → OneHotCategorical (one-hot действия)
      - DiscreteTensorSpec        → Categorical (индекс действия)
      - BoundedTensorSpec (Box)   → TanhNormal с рескейлом в [low, high]

    TODO:
      - MultiDiscreteTensorSpec: можно собрать как набор независимых Categorical
        с последующим конкатом/стеком — не реализовано в этой базовой версии.
    """
    # --- Дискретные (one-hot) действия ---
    if isinstance(action_spec, OneHotDiscreteTensorSpec):
        td_module = TensorDictModule(
            network, in_keys=["observation"], out_keys=["logits"]
        )
        return ProbabilisticActor(
            module=td_module,
            in_keys=["logits"],
            distribution_class=OneHotCategorical,
            return_log_prob=return_log_prob,
            spec=action_spec,
        )

    # --- Дискретные (индекс действия) ---
    if isinstance(action_spec, DiscreteTensorSpec):
        td_module = TensorDictModule(
            network, in_keys=["observation"], out_keys=["logits"]
        )
        return ProbabilisticActor(
            module=td_module,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=return_log_prob,
            spec=action_spec,
        )

    # --- MultiDiscrete: базовая версия не реализует ---
    if isinstance(action_spec, MultiDiscreteTensorSpec):
        raise NotImplementedError(
            "Поддержка MultiDiscrete не реализована в базовой версии. "
            "Её можно добавить как набор независимых Categorical по каждой оси."
        )

    # --- Непрерывные действия (ограниченные low/high) ---
    if isinstance(action_spec, BoundedTensorSpec):
        net_with_extractor = nn.Sequential(network, NormalParamExtractor())
        td_module = TensorDictModule(
            net_with_extractor, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        return ProbabilisticActor(
            module=td_module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": action_spec.low, "high": action_spec.high},
            return_log_prob=return_log_prob,
            spec=action_spec,
            # для eval по умолчанию даём детерминизм (среднее), можно переопределить set_exploration_type(...)
            default_interaction_type=InteractionType.DETERMINISTIC,
        )

    raise TypeError(f"Неизвестный тип action_spec: {type(action_spec)}")


# def build_stochastic_actor(network, action_spec, return_log_prob=True):
#     # Дискретные (one-hot) действия
#     if isinstance(action_spec, OneHotDiscreteTensorSpec):
#         td_module = TensorDictModule(network, in_keys=["observation"], out_keys=["logits"])
#         return ProbabilisticActor(
#             module=td_module,
#             in_keys=["logits"],
#             distribution_class=OneHotCategorical,
#             return_log_prob=return_log_prob,
#             spec=action_spec,
#         )
#     # Непрерывные действия (ограниченные low/high)
#     elif isinstance(action_spec, BoundedTensorSpec):
#         net_with_extractor = nn.Sequential(network, NormalParamExtractor())
#         td_module = TensorDictModule(net_with_extractor, in_keys=["observation"], out_keys=["loc", "scale"])
#         return ProbabilisticActor(
#             module=td_module,
#             in_keys=["loc", "scale"],
#             distribution_class=TanhNormal,
#             return_log_prob=return_log_prob,
#             distribution_kwargs={"low": action_spec.low, "high": action_spec.high},
#             spec=action_spec,
#             default_interaction_type=InteractionType.DETERMINISTIC,
#         )
#     else:
#         raise TypeError(f"Неизвестный тип action_spec: {type(action_spec)}")
    
