"""
PettingZoo backend for AgentsLab.

Создание TorchRL PettingZooEnv с применением трансформов.

Example:
    >>> from agentslab.envs.pettingzoo import make_env
    >>> from agentslab.envs.configs import PettingZooEnvConfig
    >>>
    >>> config = PettingZooEnvConfig(task="mpe/simple_spread_v3")
    >>> env = make_env(config)
"""

from __future__ import annotations

from typing import Any, List, Optional

from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.libs.pettingzoo import PettingZooEnv

from agentslab.envs.configs import PettingZooEnvConfig
from agentslab.envs.transforms import TransformBundle, build_transforms, init_observation_norm


__all__ = ["make_env", "render"]


def _get_group_names(env: PettingZooEnv) -> List[str]:
    """
    Извлекает имена групп агентов из PettingZooEnv.

    Returns:
        Список имён групп, например ["agents"] или ["team_0", "team_1"].
    """
    group_map = getattr(env, "group_map", None)

    if isinstance(group_map, dict) and group_map:
        return list(group_map.keys())

    # Fallback: стандартная группа
    return ["agents"]


def _get_marl_keys(
    env: PettingZooEnv,
    groups: List[str],
) -> tuple[List[tuple], List[tuple], List[tuple]]:
    """
    Формирует ключи для MARL трансформов.

    Returns:
        (obs_keys, reward_keys, reward_out_keys)
    """
    obs_keys = [(g, "observation") for g in groups]
    reward_keys = [(g, "reward") for g in groups]
    reward_out_keys = [(g, "episode_reward") for g in groups]

    return obs_keys, reward_keys, reward_out_keys


def _make_transforms(config: PettingZooEnvConfig, env: EnvBase) -> TransformBundle:
    """
    Создаёт трансформы для PettingZoo среды.

    Args:
        config: Конфигурация среды.
        env: Базовая PettingZooEnv (до оборачивания в TransformedEnv).

    Returns:
        TransformBundle с трансформами.
    """
    groups = _get_group_names(env)
    obs_keys, reward_keys, _ = _get_marl_keys(env, groups)

    # ВАЖНО: reward_out_keys не задаём вручную — пусть build_transforms
    # сам сформирует out_keys на основе TransformConfig.reward_sum_key.
    return build_transforms(
        config.transforms,
        obs_keys=obs_keys,
        reward_keys=reward_keys,
    )


def make_env(config: PettingZooEnvConfig) -> EnvBase:
    """
    Создаёт PettingZoo среду из конфигурации.

    Args:
        config: Конфигурация PettingZoo среды.

    Returns:
        TorchRL EnvBase (возможно обёрнутый в TransformedEnv).

    Raises:
        ImportError: Если pettingzoo не установлен.

    Example:
        >>> config = PettingZooEnvConfig(
        ...     task="mpe/simple_spread_v3",
        ...     parallel=True,
        ...     group_map="all",
        ... )
        >>> env = make_env(config)
    """
    # Подготовка kwargs
    env_kwargs = dict(config.env_kwargs)
    if config.render_mode is not None:
        env_kwargs.setdefault("render_mode", config.render_mode)

    # Создание базовой среды
    base_env = PettingZooEnv(
        task=config.task,
        parallel=config.parallel,
        return_state=config.return_state,
        use_mask=config.use_mask,
        group_map=config.group_map,
        device=config.device,
        **env_kwargs,
    )

    # Seed
    if config.seed is not None:
        base_env.set_seed(config.seed)

    # Трансформы
    bundle = _make_transforms(config, base_env)
    compose = bundle.as_compose()

    if compose is None:
        return base_env

    env = TransformedEnv(base_env, compose)

    # Инициализация ObservationNorm
    obs_cfg = config.transforms.observation_norm
    if obs_cfg is not None and bundle.observation_norm is not None:
        init_observation_norm(env, obs_cfg)

    return env


def render(
    env: EnvBase,
    *,
    mode: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Рендерит текущее состояние среды.

    Args:
        env: PettingZoo среда.
        mode: Режим рендеринга (если поддерживается).
        **kwargs: Дополнительные аргументы.

    Returns:
        Результат рендеринга (зависит от режима).

    Raises:
        RuntimeError: Если среда не поддерживает рендеринг.
    """
    if not hasattr(env, "render"):
        raise RuntimeError(
            f"Среда {type(env).__name__} не поддерживает render(). "
            "Убедитесь, что render_mode задан при создании."
        )

    if mode is not None:
        kwargs["mode"] = mode

    return env.render(**kwargs) if kwargs else env.render()
