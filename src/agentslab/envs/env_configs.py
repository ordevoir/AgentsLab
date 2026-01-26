"""
Конфигурации сред для AgentsLab (исправленная версия).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, replace
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch

# Импортируем общий резолвер из configs.py
from agentslab.core.configs import _resolve_device


T = TypeVar("T", bound="BaseEnvConfig")

NestedKey = Union[str, Tuple[str, ...]]
MarlGroupMapType = Union[Literal["all", "agent"], Dict[str, List[str]]]

# Типизация для статистик нормализации
NormStats = Union[torch.Tensor, Dict[NestedKey, torch.Tensor], float, None]


def _normalize_nested_key(key: Any) -> NestedKey:
    """Нормализует NestedKey: list → tuple, str остаётся str."""
    if isinstance(key, str):
        return key
    if isinstance(key, (list, tuple)):
        return tuple(key)
    return key


@dataclass
class BaseEnvConfig(ABC):
    """
    Базовая конфигурация среды.
    """

    device: Union[str, torch.device, None] = "cpu"
    seed: Optional[int] = None
    max_steps: Optional[int] = None

    # Нормализация наблюдений
    normalize_obs: bool = False  # ← Изменил default на False (explicit > implicit)
    norm_obs_keys: Optional[Sequence[NestedKey]] = None
    init_norm_iter: int = 1000
    loc: NormStats = None   # ← Типизировано
    scale: NormStats = None

    def __post_init__(self) -> None:
        # ✅ Резолвим device
        self.device = _resolve_device(self.device)
        
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps должен быть > 0 или None, получено {self.max_steps}")
        if self.init_norm_iter < 0:
            raise ValueError(f"init_norm_iter должен быть >= 0")

    @property
    @abstractmethod
    def env_name(self) -> str:
        """Унифицированный идентификатор среды."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует конфигурацию в JSON-friendly словарь."""
        def jsonable(x: Any) -> Any:
            if isinstance(x, torch.device):
                return str(x)
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if isinstance(x, dict):
                return {(k if isinstance(k, str) else repr(k)): jsonable(v) 
                        for k, v in x.items()}
            if isinstance(x, tuple):
                return list(x)
            return x
        return {k: jsonable(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls: type[T], d: Dict[str, Any]) -> T:
        """Создаёт конфигурацию из словаря."""
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in allowed}
        return cls(**filtered)

    def override(self: T, **updates: Any) -> T:
        """Создаёт копию с изменёнными полями."""
        return replace(self, **updates)


@dataclass
class GymEnvConfig(BaseEnvConfig):
    """Конфигурация для Gymnasium (single-agent)."""

    env_id: str = "CartPole-v1"
    render_mode: Optional[Literal["human", "rgb_array", "ansi"]] = None
    double_to_float: bool = True
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.env_id.strip():
            raise ValueError("env_id не может быть пустым")

    @property
    def env_name(self) -> str:
        return self.env_id


@dataclass
class VMASEnvConfig(BaseEnvConfig):
    """Конфигурация для VMAS (multi-agent, vectorized)."""

    scenario: str = "navigation"
    num_envs: int = 32
    continuous_actions: bool = True
    categorical_actions: bool = True
    max_steps: Optional[int] = 100  # VMAS требует явный горизонт
    group_map: Optional[MarlGroupMapType] = None
    scenario_kwargs: Dict[str, Any] = field(default_factory=dict)

    # RewardSum transform
    sum_rewards: bool = True
    reward_sum_out_key: NestedKey = ("agents", "episode_reward")

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.scenario.strip():
            raise ValueError("scenario не может быть пустым")
        if self.num_envs < 1:
            raise ValueError(f"num_envs должен быть >= 1")
        self.reward_sum_out_key = _normalize_nested_key(self.reward_sum_out_key)

    @property
    def env_name(self) -> str:
        return f"vmas/{self.scenario}"


@dataclass
class PettingZooEnvConfig(BaseEnvConfig):
    """Конфигурация для PettingZoo (multi-agent)."""

    task: str = "mpe/simple_spread_v3"
    parallel: bool = True
    return_state: bool = False
    use_mask: bool = False
    categorical_actions: bool = True
    done_on_any: Optional[bool] = None
    render_mode: Optional[Literal["human", "rgb_array"]] = None
    group_map: Optional[MarlGroupMapType] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.task.strip():
            raise ValueError("task не может быть пустым")
        if not self.parallel and not self.use_mask:
            raise ValueError("AEC-среды требуют use_mask=True")

    @property
    def env_name(self) -> str:
        return self.task


# Type Aliases
EnvConfig = Union[GymEnvConfig, VMASEnvConfig, PettingZooEnvConfig]
SingleAgentEnvConfig = GymEnvConfig
MultiAgentEnvConfig = Union[VMASEnvConfig, PettingZooEnvConfig]

