"""Построение трансформов для TorchRL сред."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from tensordict.utils import NestedKey
from torchrl.envs import EnvBase
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    RewardSum,
    StepCounter,
    Transform,
)

from agentslab.envs.configs import ObservationNormConfig, TransformConfig


__all__ = ["build_transforms", "init_observation_norm", "TransformBundle"]


@dataclass
class TransformBundle:
    """Результат построения трансформов."""

    transforms: List[Transform]
    """Список трансформов для Compose."""

    observation_norm: Optional[ObservationNorm] = None
    """Ссылка на ObservationNorm (если создан) для последующей init_stats()."""

    def as_compose(self) -> Optional[Compose]:
        """Возвращает Compose или None если трансформов нет."""
        if not self.transforms:
            return None
        return Compose(*self.transforms)


def _suffix_from_key(key: NestedKey) -> str:
    """Возвращает суффикс ключа.

    Примеры:
      ('agents', 'episode_reward') -> 'episode_reward'
      'episode_reward' -> 'episode_reward'
    """
    return key[-1] if isinstance(key, tuple) else key


def _replace_suffix(keys: Sequence[NestedKey], new_suffix: str) -> List[NestedKey]:
    """Заменяет последний элемент nested key на ``new_suffix``.

    Примеры:
      ('agents', 'reward') -> ('agents', 'episode_reward')
      'reward' -> 'episode_reward'
    """
    out: List[NestedKey] = []
    for k in keys:
        out.append((*k[:-1], new_suffix) if isinstance(k, tuple) else new_suffix)
    return out


def build_transforms(
    config: TransformConfig,
    *,
    # Явные ключи — пользователь знает свою среду лучше
    obs_keys: Optional[Sequence[NestedKey]] = None,
    reward_keys: Optional[Sequence[NestedKey]] = None,
    reward_out_keys: Optional[Sequence[NestedKey]] = None,
) -> TransformBundle:
    """
    Строит список трансформов из конфигурации.

    Args:
        config: Конфигурация трансформов.
        obs_keys: Ключи для ObservationNorm.
            По умолчанию ["observation"] для single-agent.
        reward_keys: Ключи наград для RewardSum.
            По умолчанию ["reward"].
        reward_out_keys: Выходные ключи для RewardSum.
            По умолчанию вычисляются из reward_keys с суффиксом config.reward_sum_key.

    Returns:
        TransformBundle с трансформами и ссылкой на ObservationNorm.

    Example:
        Single-agent::

            bundle = build_transforms(config)
            env = TransformedEnv(base_env, bundle.as_compose())

        Multi-agent (VMAS/PettingZoo)::

            bundle = build_transforms(
                config,
                obs_keys=[("agents", "observation")],
                reward_keys=[("agents", "reward")],
                reward_out_keys=[("agents", "episode_reward")],
            )
    """
    transforms: List[Transform] = []
    obs_norm: Optional[ObservationNorm] = None

    # Defaults
    obs_keys = list(obs_keys) if obs_keys else ["observation"]
    reward_keys = list(reward_keys) if reward_keys else ["reward"]
    if reward_out_keys is None:
        # По умолчанию пишем episode return в тот же "контейнер", что и reward_keys,
        # меняя только суффикс на config.reward_sum_key.
        reward_out_suffix = _suffix_from_key(config.reward_sum_key)
        reward_out_keys = _replace_suffix(reward_keys, reward_out_suffix)
    else:
        reward_out_keys = list(reward_out_keys)

    # 1. DoubleToFloat (почти всегда нужен для PyTorch)
    if config.double_to_float:
        transforms.append(DoubleToFloat())

    # 2. ObservationNorm
    obs_cfg = config.observation_norm
    if obs_cfg is not None and obs_cfg.enabled:
        in_keys = list(obs_cfg.in_keys) if obs_cfg.in_keys else obs_keys
        out_keys = list(obs_cfg.out_keys) if obs_cfg.out_keys else in_keys

        obs_norm = ObservationNorm(
            loc=obs_cfg.loc,
            scale=obs_cfg.scale,
            in_keys=in_keys,
            out_keys=out_keys,
            standard_normal=obs_cfg.standard_normal,
        )
        transforms.append(obs_norm)

    # 3. StepCounter
    if config.step_counter or config.max_steps is not None:
        transforms.append(StepCounter(max_steps=config.max_steps))

    # 4. RewardSum
    if config.reward_sum:
        transforms.append(
            RewardSum(
                in_keys=reward_keys,
                out_keys=reward_out_keys,
            )
        )

    return TransformBundle(transforms=transforms, observation_norm=obs_norm)


def init_observation_norm(
    env: EnvBase,
    config: ObservationNormConfig,
    num_iter: Optional[int] = None,
) -> None:
    """
    Инициализирует статистики ObservationNorm через случайные роллауты.

    Должен вызываться ПОСЛЕ создания TransformedEnv.

    Args:
        env: TransformedEnv с ObservationNorm.
        config: Конфигурация нормализации.
        num_iter: Переопределяет config.num_iter если задан.

    Example:
        >>> env = TransformedEnv(base_env, bundle.as_compose())
        >>> if bundle.observation_norm is not None:
        ...     init_observation_norm(env, obs_config)
    """
    if not config.enabled:
        return

    # Статистики уже заданы явно
    if config.loc is not None and config.scale is not None:
        return

    n = num_iter if num_iter is not None else config.num_iter
    if n <= 0:
        return

    # TorchRL way: init_stats на ObservationNorm внутри TransformedEnv
    # env.transform — это Compose, который iterable
    for transform in env.transform:
        if isinstance(transform, ObservationNorm):
            transform.init_stats(
                num_iter=n,
                reduce_dim=config.reduce_dim,
                cat_dim=config.cat_dim,
            )
            break
