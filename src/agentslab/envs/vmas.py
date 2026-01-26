"""
VMAS backend for AgentsLab.

Provides factory and utilities for VMAS multi-agent environments.
"""

from __future__ import annotations

from typing import Any
from dataclasses import replace

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
    tcfg = config.transforms
    if tcfg.step_counter and tcfg.max_steps is None:
        tcfg = replace(tcfg, max_steps=config.max_steps)

    # VMAS предоставляет эти атрибуты (fallback на стандартные)
    obs_keys = getattr(env, "observation_keys", [("agents", "observation")])
    reward_keys = getattr(env, "reward_keys", [("agents", "reward")])

    # reward_out_keys вычисляется внутри build_transforms на основе tcfg.reward_sum_key
    bundle = build_transforms(
        tcfg,
        obs_keys=list(obs_keys),
        reward_keys=list(reward_keys),
    )

    # 4) Wrap если есть трансформы
    if bundle.transforms:
        env = TransformedEnv(env, bundle.as_compose())

        # Init observation norm stats
        if bundle.observation_norm and tcfg.observation_norm:
            init_observation_norm(env, tcfg.observation_norm)

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
