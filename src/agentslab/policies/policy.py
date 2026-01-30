from functools import partial
from typing import List, Sequence, Union

import torch
import torch.nn as nn
from torch.distributions import Distribution, OneHotCategorical, Categorical

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.data.tensor_specs import (
    OneHotDiscreteTensorSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,  # deprecated, но нужен для обратной совместимости
    BoundedTensorSpec,
)

# TorchRL 0.9+ переименовал MultiDiscreteTensorSpec в MultiCategorical
# Импортируем для совместимости с будущими версиями
try:
    from torchrl.data.tensor_specs import MultiCategorical as MultiCategoricalSpec
except ImportError:
    MultiCategoricalSpec = MultiDiscreteTensorSpec


__all__ = [
    "MultiCategorical",
    "get_num_action_logits",
    "build_stochastic_actor",
]


# =============================================================================
# MultiCategorical Distribution
# =============================================================================

class MultiCategorical(Distribution):
    """
    Распределение для MultiDiscrete action space.

    Оборачивает несколько независимых Categorical распределений.
    Каждое под-действие семплируется независимо, log_prob и entropy
    суммируются по всем под-действиям.

    Args:
        nvec: размерности каждого под-действия [n1, n2, ...].
            Например, [3, 5, 4] означает 3 под-действия с 3, 5 и 4 вариантами.
        logits: тензор логитов формы (..., sum(nvec)).
            Логиты конкатенированы: первые n1 для первого действия,
            следующие n2 для второго и т.д.
        probs: альтернатива logits — тензор вероятностей той же формы.
            Указывать либо logits, либо probs, но не оба.

    Example:
        >>> nvec = [3, 5, 4]  # 3 под-действия
        >>> logits = torch.randn(32, sum(nvec))  # batch_size=32
        >>> dist = MultiCategorical(nvec, logits=logits)
        >>> actions = dist.sample()  # shape: (32, 3)
        >>> log_p = dist.log_prob(actions)  # shape: (32,)
        >>> ent = dist.entropy()  # shape: (32,)

    Note:
        Для независимых под-действий a = (a_1, ..., a_k):

        - Совместная вероятность: π(a|s) = ∏ π_i(a_i|s)
        - Log prob: log π(a|s) = Σ log π_i(a_i|s)
        - Энтропия: H[π] = Σ H[π_i]

    See Also:
        - PyTorch Issue #43250: https://github.com/pytorch/pytorch/issues/43250
        - Stable Baselines3 MultiCategoricalDistribution
    """

    arg_constraints = {}

    def __init__(
        self,
        nvec: Sequence[int],
        logits: torch.Tensor = None,
        probs: torch.Tensor = None,
        validate_args: bool = None,
    ):
        self.nvec = list(nvec)
        self.n_cats = len(self.nvec)

        if (logits is None) == (probs is None):
            raise ValueError("Укажите либо logits, либо probs, но не оба")

        params = logits if logits is not None else probs
        param_key = "logits" if logits is not None else "probs"

        # Проверяем размерность
        expected_size = sum(self.nvec)
        if params.shape[-1] != expected_size:
            raise ValueError(
                f"Последняя размерность {param_key} должна быть {expected_size} "
                f"(sum of nvec={self.nvec}), получено {params.shape[-1]}"
            )

        # Создаём список Categorical распределений
        self.cats: List[Categorical] = []
        offset = 0
        for n in self.nvec:
            cat_params = params[..., offset:offset + n]
            self.cats.append(Categorical(**{param_key: cat_params}, validate_args=validate_args))
            offset += n

        # batch_shape из первого распределения
        batch_shape = self.cats[0].batch_shape
        event_shape = torch.Size([self.n_cats])

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Семплирует действия из всех под-распределений.

        Returns:
            Тензор формы (*sample_shape, *batch_shape, n_cats) с индексами действий.
        """
        samples = [cat.sample(sample_shape) for cat in self.cats]
        return torch.stack(samples, dim=-1)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Reparametrized sample не поддерживается для дискретных распределений.
        """
        raise NotImplementedError(
            "rsample не поддерживается для дискретных распределений. "
            "Используйте sample() или Straight-Through Estimator."
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет log probability для заданных действий.

        Args:
            value: тензор действий формы (..., n_cats).

        Returns:
            Сумма log prob по всем под-действиям, форма (...).
        """
        if value.shape[-1] != self.n_cats:
            raise ValueError(
                f"Последняя размерность value должна быть {self.n_cats}, "
                f"получено {value.shape[-1]}"
            )

        log_probs = []
        for i, cat in enumerate(self.cats):
            log_probs.append(cat.log_prob(value[..., i]))

        return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """
        Вычисляет суммарную энтропию всех под-распределений.

        Returns:
            Сумма энтропий, форма (*batch_shape).
        """
        entropies = [cat.entropy() for cat in self.cats]
        return torch.stack(entropies, dim=-1).sum(dim=-1)

    @property
    def probs(self) -> List[torch.Tensor]:
        """Список вероятностей для каждого под-распределения."""
        return [cat.probs for cat in self.cats]

    @property
    def logits(self) -> List[torch.Tensor]:
        """Список логитов для каждого под-распределения."""
        return [cat.logits for cat in self.cats]

    @property
    def mode(self) -> torch.Tensor:
        """Наиболее вероятные действия (argmax) для каждого под-распределения."""
        modes = [cat.probs.argmax(dim=-1) for cat in self.cats]
        return torch.stack(modes, dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """
        Для дискретных распределений mean возвращает mode.

        Это нужно для совместимости с TorchRL ProbabilisticActor,
        который использует mean как fallback для deterministic_sample.
        """
        return self.mode

    def __repr__(self) -> str:
        return f"MultiCategorical(nvec={self.nvec})"


# =============================================================================
# Utility Functions
# =============================================================================

def get_num_action_logits(action_spec) -> int:
    """
    Возвращает число логитов, которое должна выпускать политика до преобразования
    в распределение (удобно для конфигурации выходного слоя MLP).

    Args:
        action_spec: спецификация действий из TorchRL.

    Returns:
        Число логитов для выходного слоя сети.

    Note:
        - OneHotDiscrete / Discrete: n (число категорий)
        - MultiDiscrete: sum(nvec) (суммарное число категорий)
        - Bounded (Box): размерность действия
          (для непрерывного актора умножьте на 2 для μ и σ)

    Example:
        >>> from torchrl.data import DiscreteTensorSpec, MultiDiscreteTensorSpec
        >>> get_num_action_logits(DiscreteTensorSpec(n=5))
        5
        >>> get_num_action_logits(MultiDiscreteTensorSpec(nvec=[3, 4, 5]))
        12
    """
    # MultiDiscrete/MultiCategorical проверяем ДО Discrete, т.к. наследуется от Discrete
    if isinstance(action_spec, (MultiDiscreteTensorSpec, MultiCategoricalSpec)):
        return int(action_spec.nvec.sum())
    if isinstance(action_spec, OneHotDiscreteTensorSpec):
        return int(action_spec.n)
    if isinstance(action_spec, DiscreteTensorSpec):
        return int(action_spec.n)
    if isinstance(action_spec, BoundedTensorSpec):
        return int(action_spec.shape[-1])
    raise TypeError(f"Неизвестный тип action_spec: {type(action_spec)}")


# =============================================================================
# Actor Builder
# =============================================================================

def build_stochastic_actor(
    network: nn.Module,
    action_spec,
    return_log_prob: bool = True,
    in_keys: Sequence[str] = ("observation",),
):
    """
    Строит стохастического актёра под тип action_spec.

    Поддерживаемые типы action_spec:
        - OneHotDiscreteTensorSpec → OneHotCategorical
            logits → softmax → sample one-hot vector
        - DiscreteTensorSpec → Categorical
            logits → softmax → sample index
        - MultiDiscreteTensorSpec / MultiCategorical → MultiCategorical
            logits → split by nvec → k independent softmax → sample k indices
        - BoundedTensorSpec (Box) → TanhNormal
            logits → NormalParamExtractor(μ, σ) → sample u ~ N(μ, σ) → tanh(u) → scale to [low, high]

    Args:
        network: нейронная сеть, принимающая observation и выдающая логиты.
            Число выходов должно соответствовать get_num_action_logits(action_spec)
            (для непрерывных действий — удвоенное, т.к. NormalParamExtractor
            добавляется автоматически).
        action_spec: спецификация действий из среды (env.action_spec).
        return_log_prob: возвращать ли log_prob в TensorDict (default: True).
        in_keys: входные ключи для TensorDictModule (default: ["observation"]).

    Returns:
        ProbabilisticActor, готовый к использованию с TorchRL.

    Example:
        >>> from torchrl.envs import GymEnv
        >>> env = GymEnv("CartPole-v1")
        >>> obs_dim = env.observation_spec["observation"].shape[-1]
        >>> act_dim = get_num_action_logits(env.action_spec)
        >>> network = nn.Sequential(
        ...     nn.Linear(obs_dim, 64), nn.ReLU(),
        ...     nn.Linear(64, act_dim)
        ... )
        >>> actor = build_stochastic_actor(network, env.action_spec)

    See Also:
        - get_num_action_logits: для определения размера выходного слоя
        - MultiCategorical: для деталей реализации multi-discrete
    """
    in_keys = list(in_keys)

    # --- MultiDiscrete / MultiCategorical (несколько независимых Categorical) ---
    # Проверяем ДО DiscreteTensorSpec, т.к. MultiDiscrete наследуется от Discrete
    if isinstance(action_spec, (MultiDiscreteTensorSpec, MultiCategoricalSpec)):
        nvec = action_spec.nvec.tolist()

        td_module = TensorDictModule(
            network, in_keys=in_keys, out_keys=["logits"]
        )
        return ProbabilisticActor(
            module=td_module,
            in_keys=["logits"],
            distribution_class=partial(MultiCategorical, nvec=nvec),
            return_log_prob=return_log_prob,
            spec=action_spec,
        )

    # --- Дискретные (one-hot) действия ---
    if isinstance(action_spec, OneHotDiscreteTensorSpec):
        td_module = TensorDictModule(
            network, in_keys=in_keys, out_keys=["logits"]
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
            network, in_keys=in_keys, out_keys=["logits"]
        )
        return ProbabilisticActor(
            module=td_module,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=return_log_prob,
            spec=action_spec,
        )

    # --- Непрерывные действия (ограниченные low/high) ---
    if isinstance(action_spec, BoundedTensorSpec):
        net_with_extractor = nn.Sequential(network, NormalParamExtractor())
        td_module = TensorDictModule(
            net_with_extractor, in_keys=in_keys, out_keys=["loc", "scale"]
        )
        return ProbabilisticActor(
            module=td_module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": action_spec.low, "high": action_spec.high},
            return_log_prob=return_log_prob,
            spec=action_spec,
        )

    raise TypeError(f"Неизвестный тип action_spec: {type(action_spec)}")
