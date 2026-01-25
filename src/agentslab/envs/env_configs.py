"""
Конфигурации сред для AgentsLab.

Модуль предоставляет dataclass-конфигурации для различных бэкендов сред:
- GymEnvConfig: Gymnasium (single-agent)
- VMASEnvConfig: VMAS (multi-agent, vectorized)
- PettingZooEnvConfig: PettingZoo (multi-agent)
"""

from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass, field, fields, replace
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import torch

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None


T = TypeVar("T", bound="BaseEnvConfig")


__all__ = [
    "BaseEnvConfig",
    "GymEnvConfig",
    "VMASEnvConfig",
    "PettingZooEnvConfig",
    "EnvConfig",
    "SingleAgentEnvConfig",
    "MultiAgentEnvConfig",
]


# =============================================================================
# Базовая конфигурация
# =============================================================================


@dataclass
class BaseEnvConfig(ABC):
    """
    Базовая конфигурация среды.

    Не используйте напрямую — наследуйте конкретные классы.

    Attributes:
        device: Устройство ("cpu", "cuda", "cuda:N", "mps", "auto") или torch.device.
        seed: Random seed для воспроизводимости.
        max_steps: Максимальная длина эпизода (time limit).
            В AgentsLab это трактуется как *truncation по тайм-лимиту*.
            (Важно для PPO/GAE: на truncation обычно делается bootstrap value.)
        normalize_obs: Применять нормализацию наблюдений.
        init_norm_iter: Число шагов для сбора статистики нормализации.
        loc: Среднее для нормализации (если None — вычисляется автоматически).
        scale: Стандартное отклонение для нормализации.

    Notes:
        loc/scale — это скорее состояние нормализатора. Если вы задаёте их явно,
        фабрика среды должна использовать эти значения, а init_norm_iter можно игнорировать.
    """

    device: Union[str, torch.device, None] = "cpu"
    seed: Optional[int] = None
    max_steps: Optional[int] = None

    normalize_obs: bool = True
    init_norm_iter: int = 1000
    loc: Optional[Any] = None
    scale: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps должен быть > 0 или None, получено {self.max_steps}")
        if self.init_norm_iter < 0:
            raise ValueError(f"init_norm_iter должен быть >= 0, получено {self.init_norm_iter}")

    def resolve_device(self) -> torch.device:
        """
        Возвращает torch.device для использования в фабриках.

        Поддерживает:
          - строковые устройства ("cpu", "cuda", "cuda:0", "mps", ...)
          - torch.device
          - None (=> cpu)
          - "auto" (выбор лучшего доступного: cuda -> mps -> cpu)

        Замечание:
          torch.device("cuda") сам по себе не проверяет доступность CUDA.
          Проверку (и fail-fast) разумнее делать в фабрике среды/раннере.
        """
        dev = self.device
        if dev is None:
            return torch.device("cpu")
        if isinstance(dev, torch.device):
            return dev
        if isinstance(dev, str):
            s = dev.strip().lower()
            if s == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda")
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                    return torch.device("mps")
                return torch.device("cpu")
            return torch.device(dev)
        return torch.device(str(dev))

    @staticmethod
    def _jsonable(x: Any) -> Any:
        """Приводит значения к JSON-friendly виду (для to_dict())."""
        if isinstance(x, torch.device):
            return str(x)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if np is not None and isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: BaseEnvConfig._jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            # tuple -> list (JSON-friendly). Тип обратно восстановим в from_dict/__post_init__ при необходимости.
            return [BaseEnvConfig._jsonable(v) for v in x]
        return x

    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализует конфигурацию в словарь (JSON-friendly).
        """
        return self._jsonable(asdict(self))

    @classmethod
    def _coerce_from_dict(cls, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Хук для приведения типов при восстановлении из dict.
        Наследники могут переопределять.
        """
        dd = dict(d)

        # device: строка -> torch.device (кроме "auto")
        dev = dd.get("device")
        if isinstance(dev, str) and dev.strip().lower() != "auto":
            try:
                dd["device"] = torch.device(dev)
            except Exception:
                pass

        return dd

    @classmethod
    def from_dict(cls: type[T], d: Dict[str, Any]) -> T:
        """
        Создаёт конфигурацию из словаря.

        - фильтрует неизвестные ключи (совместимость между версиями)
        - выполняет приведение типов через _coerce_from_dict()
        """
        raw = cls._coerce_from_dict(d)

        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in allowed}
        return cls(**filtered)

    def override(self: T, **updates: Any) -> T:
        """Создаёт копию с изменёнными полями."""
        return replace(self, **updates)


# =============================================================================
# Single-Agent
# =============================================================================


@dataclass
class GymEnvConfig(BaseEnvConfig):
    """
    Конфигурация для Gymnasium (single-agent).

    Attributes:
        env_id: Идентификатор среды ("CartPole-v1", "Ant-v4", "ALE/Pong-v5").
        render_mode: Режим визуализации (None, "human", "rgb_array", "ansi").
        obs_keys: Ключи наблюдений для dict observation spaces.
        double_to_float: Конвертировать float64 → float32.
        env_kwargs: Дополнительные аргументы для gymnasium.make().
    """

    env_id: str = "CartPole-v1"
    render_mode: Optional[Literal["human", "rgb_array", "ansi"]] = None
    obs_keys: Optional[Sequence[str]] = None
    double_to_float: bool = True
    env_kwargs: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Multi-Agent
# =============================================================================


@dataclass
class VMASEnvConfig(BaseEnvConfig):
    """
    Конфигурация для VMAS (multi-agent, vectorized).

    VMAS — векторизованный симулятор на PyTorch с GPU-ускорением.

    Attributes:
        scenario: Название сценария ("navigation", "transport", "flocking", ...).
        num_envs: Число параллельных сред (batch size).
        continuous_actions: Непрерывные (True) или дискретные (False) действия.
        categorical_actions: Для дискретных: categorical (True) или one-hot (False).
        max_steps: Горизонт эпизода (time limit). В AgentsLab трактуется как truncation.
        group_map: Группировка агентов в TensorDict (None = авто по именам).
        scenario_kwargs: Параметры сценария (n_agents, world_size, ...).
        sum_rewards: Добавлять трансформ для суммарной награды за эпизод.
        reward_sum_out_key: Ключ (в TensorDict) для суммарной эпизодической награды.
    """

    scenario: str = "navigation"
    num_envs: int = 32
    continuous_actions: bool = True
    categorical_actions: bool = True
    max_steps: Optional[int] = 100
    group_map: Optional[Union[str, Dict[str, List[str]]]] = None
    scenario_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Для совместимости с текущей фабрикой VMAS (RewardSum)
    sum_rewards: bool = True
    reward_sum_out_key: tuple[str, str] = ("agents", "episode_reward")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_envs < 1:
            raise ValueError(f"num_envs должен быть >= 1, получено {self.num_envs}")

        # Восстановление tuple после JSON-сериализации (tuple -> list).
        if isinstance(self.reward_sum_out_key, list):
            self.reward_sum_out_key = tuple(self.reward_sum_out_key)  # type: ignore[assignment]

        # Лёгкая валидация ключа
        if not (isinstance(self.reward_sum_out_key, tuple) and len(self.reward_sum_out_key) == 2):
            raise ValueError(
                f"reward_sum_out_key должен быть tuple[str, str] длины 2, получено {self.reward_sum_out_key!r}"
            )

    @classmethod
    def _coerce_from_dict(cls, d: Dict[str, Any]) -> Dict[str, Any]:
        dd = super()._coerce_from_dict(d)

        # reward_sum_out_key: list -> tuple
        rkey = dd.get("reward_sum_out_key")
        if isinstance(rkey, list):
            dd["reward_sum_out_key"] = tuple(rkey)

        return dd


@dataclass
class PettingZooEnvConfig(BaseEnvConfig):
    """
    Конфигурация для PettingZoo (multi-agent).

    Поддерживает Parallel API (одновременные действия) и AEC (пошаговые).

    Attributes:
        task: Идентификатор задачи ("mpe/simple_spread_v3", "sisl/multiwalker_v9").
        parallel: True = ParallelEnv, False = AECEnv.
        return_state: Возвращать глобальное состояние (для centralized critic).
        use_mask: Выводить маски агентов. Обязательно для AEC.
        categorical_actions: Для дискретных: categorical (True) или one-hot (False).
        done_on_any: Завершение при done любого (True) или всех (False) агентов.
        render_mode: Режим визуализации (None, "human", "rgb_array").
        group_map: Группировка агентов в TensorDict.
        env_kwargs: Дополнительные параметры среды.
    """

    task: str = "mpe/simple_spread_v3"
    parallel: bool = True
    return_state: bool = False
    use_mask: bool = False
    categorical_actions: bool = True
    done_on_any: Optional[bool] = None
    render_mode: Optional[Literal["human", "rgb_array"]] = None
    group_map: Optional[Union[str, Dict[str, List[str]]]] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.parallel and not self.use_mask:
            import warnings

            warnings.warn(
                "AEC-среды (parallel=False) требуют use_mask=True",
                UserWarning,
            )


# =============================================================================
# Type Aliases
# =============================================================================

EnvConfig = Union[GymEnvConfig, VMASEnvConfig, PettingZooEnvConfig]
SingleAgentEnvConfig = GymEnvConfig
MultiAgentEnvConfig = Union[VMASEnvConfig, PettingZooEnvConfig]
