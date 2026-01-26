"""
AgentsLab Environments.

Модуль предоставляет унифицированный интерфейс для работы с различными
средами обучения с подкреплением.

Example:
    >>> from agentslab.envs import GymEnvConfig, make_env
    >>> config = GymEnvConfig(env_name="CartPole-v1")
    >>> env = make_env(config)
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torchrl.envs import EnvBase

from agentslab.envs.configs import (
    # Конфиги всегда доступны
    GymEnvConfig,
    VMASEnvConfig,
    PettingZooEnvConfig,
    TransformConfig,
    ObservationNormConfig,
    EnvConfig,
)

__all__ = [
    # Configs
    "GymEnvConfig",
    "VMASEnvConfig",
    "PettingZooEnvConfig",
    "TransformConfig",
    "ObservationNormConfig",
    "EnvConfig",
    # Factory (universal)
    "make_env",
    # Lazy submodules
    "gymnasium",
    "vmas",
    "pettingzoo",
    "transforms",
]


def make_env(config: EnvConfig) -> "EnvBase":
    """
    Универсальная фабрика сред.
    
    Автоматически выбирает правильный бэкенд по типу конфига.
    
    Args:
        config: Конфигурация среды.
        
    Returns:
        TorchRL EnvBase.
        
    Example:
        >>> env = make_env(GymEnvConfig(env_name="CartPole-v1"))
        >>> env = make_env(VMASEnvConfig(scenario="navigation"))
    """
    if isinstance(config, GymEnvConfig):
        from agentslab.envs.gymnasium import make_env as _make
        return _make(config)
    
    if isinstance(config, VMASEnvConfig):
        from agentslab.envs.vmas import make_env as _make
        return _make(config)
    
    if isinstance(config, PettingZooEnvConfig):
        from agentslab.envs.pettingzoo import make_env as _make
        return _make(config)
    
    raise TypeError(f"Unknown config type: {type(config)}")



def __getattr__(name: str):
    # Ленивый импорт модулей (только при запросе)

    import importlib
    if name in {"gymnasium", "vmas", "pettingzoo", "transforms"}:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # кеш, чтобы повторно не вызывать __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

