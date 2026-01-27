"""
VMAS backend for AgentsLab.

Provides factory and utilities for VMAS multi-agent environments.
"""

from __future__ import annotations

from typing import Any

from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv

from agentslab.envs.configs import VMASEnvConfig
from agentslab.envs.transforms import build_transforms, init_observation_norm

__all__ = ["make_env", "render"]


def make_env(config: VMASEnvConfig) -> EnvBase:
    """
    Создаёт VMAS среду из конфигурации.

    Args:
        config: Конфигурация VMAS среды.

    Returns:
        EnvBase (возможно обёрнутый в TransformedEnv).

    Example:
        >>> config = VMASEnvConfig(scenario="navigation", num_envs=64)
        >>> env = make_env(config)
    """
    # 1) Создаём базовую среду
    env = VmasEnv(
        scenario=config.scenario,
        num_envs=config.num_envs,
        device=config.device,
        continuous_actions=config.continuous_actions,
        max_steps=config.max_steps,
        group_map=config.group_map,
        **config.scenario_kwargs,
    )

    # 2) Seed
    if config.seed is not None:
        env.set_seed(config.seed)

    # 3) Transforms
    # NOTE: max_steps обрабатывается нативно VmasEnv (эффективнее, чем StepCounter).
    # Если нужен StepCounter — задайте TransformConfig.max_steps явно.
    obs_keys = getattr(env, "observation_keys", [("agents", "observation")])
    reward_keys = getattr(env, "reward_keys", [("agents", "reward")])

    bundle = build_transforms(
        config.transforms,
        obs_keys=list(obs_keys),
        reward_keys=list(reward_keys),
    )

    # 4) Wrap если есть трансформы
    if bundle.transforms:
        env = TransformedEnv(env, bundle.as_compose())

        # Init observation norm stats
        if bundle.observation_norm and config.transforms.observation_norm:
            init_observation_norm(env, config.transforms.observation_norm)

    return env


def render(env: EnvBase, **kwargs: Any) -> Any:
    """
    Рендерит среду.

    Args:
        env: VMAS среда.
        **kwargs: Аргументы для env.render().

    Returns:
        Результат рендеринга (зависит от render_mode).
    """
    return env.render(**kwargs)
