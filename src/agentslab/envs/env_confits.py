"""
Конфигурации сред для AgentsLab.

Модуль предоставляет dataclass-конфигурации для различных бэкендов сред:
- GymEnvConfig: Gymnasium (single-agent)
- VMASEnvConfig: VMAS (multi-agent, vectorized)
- PettingZooEnvConfig: PettingZoo (multi-agent)
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, asdict, replace
import torch
from typing import (
    Optional, 
    Union, 
    Sequence, 
    Dict, 
    Any, 
    List,
    Literal,
    TypeVar,
)


T = TypeVar("T", bound="BaseEnvConfig")


# =============================================================================
# Базовая конфигурация
# =============================================================================

@dataclass
class BaseEnvConfig(ABC):
    """
    Базовая конфигурация среды.
    
    Не используйте напрямую — наследуйте конкретные классы.
    
    Attributes:
        device: Устройство ("cpu", "cuda", "cuda:N", "mps", "auto").
        seed: Random seed для воспроизводимости.
        max_steps: Максимальная длина эпизода (truncation).
        normalize_obs: Применять нормализацию наблюдений.
        init_norm_iter: Число шагов для сбора статистики нормализации.
        loc: Среднее для нормализации (если None — вычисляется автоматически).
        scale: Стандартное отклонение для нормализации.
    """
    
    device: Union[str, torch.device, None] = "cpu"
    seed: Optional[int] = None
    max_steps: Optional[int] = None
    
    normalize_obs: bool = True
    init_norm_iter: int = 1000
    loc: Optional[Union[float, Sequence[float], Any]] = None
    scale: Optional[Union[float, Sequence[float], Any]] = None
    
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализует конфигурацию в словарь.
        
        Returns:
            Dict со всеми полями конфигурации.
        """
        d = asdict(self)
        for key in ("loc", "scale"):
            if isinstance(d.get(key), torch.Tensor):
                d[key] = d[key].tolist()
        return d
    
    @classmethod
    def from_dict(cls: type[T], d: Dict[str, Any]) -> T:
        """
        Создаёт конфигурацию из словаря.
        
        Args:
            d: Словарь с полями конфигурации.
        
        Returns:
            Новый экземпляр конфигурации.
        """
        return cls(**d)
    
    def override(self: T, **updates: Any) -> T:
        """
        Создаёт копию с изменёнными полями.
        
        Args:
            **updates: Поля для обновления.
        
        Returns:
            Новый экземпляр с обновлёнными полями.
        """
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
        max_steps: Горизонт эпизода. Устанавливает terminated, не truncated.
        group_map: Группировка агентов в TensorDict (None = авто по именам).
        scenario_kwargs: Параметры сценария (n_agents, world_size, ...).
    """
    
    scenario: str = "navigation"
    num_envs: int = 32
    continuous_actions: bool = True
    categorical_actions: bool = True
    max_steps: Optional[int] = 100
    group_map: Optional[Union[str, Dict[str, List[str]]]] = None
    scenario_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.num_envs < 1:
            raise ValueError(f"num_envs должен быть >= 1, получено {self.num_envs}")


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
    
    def __post_init__(self):
        if not self.parallel and not self.use_mask:
            import warnings
            warnings.warn(
                "AEC-среды (parallel=False) требуют use_mask=True",
                UserWarning
            )


# =============================================================================
# Type Aliases
# =============================================================================

EnvConfig = Union[GymEnvConfig, VMASEnvConfig, PettingZooEnvConfig]
SingleAgentEnvConfig = GymEnvConfig
MultiAgentEnvConfig = Union[VMASEnvConfig, PettingZooEnvConfig]


__all__ = [
    "BaseEnvConfig",
    "GymEnvConfig",
    "VMASEnvConfig",
    "PettingZooEnvConfig",
    "EnvConfig",
    "SingleAgentEnvConfig",
    "MultiAgentEnvConfig",
]
