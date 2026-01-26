"""
Gymnasium backend for AgentsLab.

Создаёт TorchRL GymEnv и применяет transforms через общий builder.

Public API:
    - make_env(config: GymEnvConfig) -> EnvBase
    - render(env, ...) -> Any
"""
from __future__ import annotations

from typing import Any, List, Optional
import warnings

from torchrl.envs import EnvBase, GymEnv, TransformedEnv

from agentslab.envs.configs import GymEnvConfig
from agentslab.envs.transforms import (
    build_transforms,
    init_observation_norm,
)


__all__ = ["make_env", "render"]


# ============================================================================
# Constants
# ============================================================================

# Стандартные ключи для single-agent Gymnasium сред
_DEFAULT_OBS_KEYS: List[str] = ["observation"]
_DEFAULT_REWARD_KEYS: List[str] = ["reward"]


# ============================================================================
# Helpers
# ============================================================================

def _parse_batch_size(batch_size: Optional[Any]) -> int:
    """
    Извлекает num_envs из batch_size.

    Args:
        batch_size: None, torch.Size, или sequence.

    Returns:
        0 если batch_size пустой, иначе первый элемент.

    Raises:
        ValueError: Если batch_size имеет больше одного измерения.
    """
    if batch_size is None or len(batch_size) == 0:
        return 0

    if len(batch_size) != 1:
        raise ValueError(
            f"GymEnvConfig.batch_size должен быть None или [N], "
            f"получено: {batch_size}"
        )

    n = int(batch_size[0])
    if n <= 0:
        raise ValueError(f"batch_size[0] должен быть > 0, получено: {n}")

    return n


# ============================================================================
# Public API
# ============================================================================

def make_env(config: GymEnvConfig) -> EnvBase:
    """
    Создаёт Gymnasium среду из конфигурации.

    Args:
        config: Конфигурация среды.

    Returns:
        TorchRL EnvBase (GymEnv или TransformedEnv).

    Example:
        >>> config = GymEnvConfig(env_name="CartPole-v1", device="cuda")
        >>> env = make_env(config)
        >>> td = env.reset()
    """
    # 1. Собираем kwargs для GymEnv
    gym_kwargs = dict(config.gym_kwargs or {})

    if config.render_mode is not None:
        gym_kwargs.setdefault("render_mode", config.render_mode)

    # Векторизация
    num_envs = _parse_batch_size(config.batch_size)
    if num_envs > 0:
        if "num_envs" in gym_kwargs:
            warnings.warn(
                "GymEnvConfig.batch_size задан, но gym_kwargs содержит num_envs — "
                "используется значение из batch_size",
                UserWarning,
                stacklevel=2,
            )
        gym_kwargs["num_envs"] = num_envs

    # Frame skip (всегда передаём если != 1)
    if config.frame_skip != 1:
        gym_kwargs["frame_skip"] = config.frame_skip

    # 2. Создаём базовую среду
    env = GymEnv(
        env_name=config.env_name,
        device=config.device,
        categorical_action_encoding=config.categorical_action_encoding,
        **gym_kwargs,
    )

    # 3. Seed
    if config.seed is not None:
        env.set_seed(config.seed)

    # 4. Трансформы
    # ВАЖНО: reward_out_keys не задаём вручную — пусть build_transforms
    # сам сформирует out_keys на основе TransformConfig.reward_sum_key.
    bundle = build_transforms(
        config.transforms,
        obs_keys=_DEFAULT_OBS_KEYS,
        reward_keys=_DEFAULT_REWARD_KEYS,
    )

    if bundle.transforms:
        env = TransformedEnv(env, bundle.as_compose())

        # Init ObservationNorm stats
        obs_cfg = config.transforms.observation_norm
        if obs_cfg is not None and bundle.observation_norm is not None:
            init_observation_norm(env, obs_cfg)

    return env


def render(env: EnvBase, *args: Any, **kwargs: Any) -> Any:
    """
    Рендерит среду.

    Args:
        env: TorchRL среда (GymEnv или TransformedEnv).
        *args, **kwargs: Передаются в env.render().

    Returns:
        Результат рендеринга (зависит от render_mode).

    Raises:
        RuntimeError: Если render недоступен.
    """
    # TransformedEnv и GymEnv имеют render()
    if hasattr(env, "render"):
        return env.render(*args, **kwargs)

    # Fallback: ищем базовый Gymnasium env
    base = env
    while hasattr(base, "base_env"):
        base = base.base_env

    gym_env = getattr(base, "_env", None)
    if gym_env is not None and hasattr(gym_env, "render"):
        return gym_env.render(*args, **kwargs)

    raise RuntimeError(
        f"render() недоступен для {type(env).__name__}. "
        f"Убедитесь, что render_mode задан при создании среды."
    )
