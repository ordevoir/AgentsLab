
"""
Gymnasium backend (single-agent) for AgentsLab.

Содержит:
- make_gym_env: фабрика TorchRL GymEnv + transforms (dtype, obs-norm, step counter)
- play: прокат одного эпизода с рендером (rgb_array/human)
- is_acts_discrete: утилита для определения дискретности action_spec
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
)
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import Compose, DoubleToFloat, ObservationNorm, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from agentslab.envs.env_configs import GymEnvConfig

NestedKey = Union[str, Tuple[str, ...]]  # приблизительно; TorchRL/TensorDict допускают nested ключи


def is_acts_discrete(action_spec) -> bool:
    """True для всех дискретных вариантов: OneHotDiscreteTensorSpec, DiscreteTensorSpec, MultiDiscreteTensorSpec."""
    return isinstance(action_spec, (OneHotDiscreteTensorSpec, DiscreteTensorSpec, MultiDiscreteTensorSpec))


def _infer_float_obs_keys(observation_spec) -> List[NestedKey]:
    """
    Выбираем float-наблюдения (не пиксели).
    Возвращаем список ключей (NestedKey) для CompositeSpec или ["observation"] для простого spec.
    """
    deny_substrings = ("pixel", "pixels", "image", "images", "rgb", "frame", "frames")
    float_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}

    try:
        if hasattr(observation_spec, "keys"):
            keys: List[NestedKey] = []
            for k in observation_spec.keys(True, True):
                sub = observation_spec[k]
                dt = getattr(sub, "dtype", None)
                if dt in float_dtypes:
                    name = str(k).lower()
                    if any(s in name for s in deny_substrings):
                        continue
                    keys.append(k)
            return keys or ["observation"]

        dt = getattr(observation_spec, "dtype", None)
        if dt in float_dtypes:
            return ["observation"]
        return ["observation"]
    except Exception:
        return ["observation"]


def make_gym_env(cfg: GymEnvConfig) -> TransformedEnv:
    """
    Создаёт TorchRL GymEnv и оборачивает трансформами согласно GymEnvConfig.

    Соответствие GymEnvConfig:
      - env_id -> GymEnv(env_name=...)
      - render_mode -> GymEnv(render_mode=...)
      - env_kwargs -> пробрасывается в GymEnv / gymnasium.make
      - seed -> base_env.set_seed
      - double_to_float -> DoubleToFloat
      - normalize_obs/init_norm_iter/loc/scale/obs_keys -> ObservationNorm (+ init_stats / ручные loc/scale)
      - max_steps -> StepCounter(max_steps=...) (truncation)
    """
    # Важно: по твоему условию device уже резолвится в общем конфиге,
    # поэтому здесь ожидаем, что cfg.device — уже валидное устройство/строка.
    device = cfg.device

    env_kwargs: Dict[str, Any] = dict(getattr(cfg, "env_kwargs", {}) or {})
    # Не допускаем конфликтов по именам с явными аргументами.
    # (Если user положил render_mode в env_kwargs — cfg.render_mode имеет приоритет.)
    env_kwargs.pop("render_mode", None)
    env_kwargs.pop("device", None)
    env_kwargs.pop("env_name", None)

    # GymEnv в разных версиях TorchRL принимал kwargs по-разному.
    # Делаем безопасно: сначала пробуем прокинуть как **env_kwargs, при TypeError — как env_kwargs=...
    try:
        base_env = GymEnv(
            env_name=cfg.env_id,
            device=device,
            render_mode=cfg.render_mode,
            **env_kwargs,
        )
    except TypeError:
        base_env = GymEnv(
            env_name=cfg.env_id,
            device=device,
            render_mode=cfg.render_mode,
            env_kwargs=env_kwargs,
        )

    # Сидирование TorchRL-окружения
    if cfg.seed is not None:
        base_env.set_seed(cfg.seed)

    transforms: List[Any] = []

    # dtype: float64 -> float32
    if cfg.double_to_float:
        transforms.append(DoubleToFloat())

    # Нормализация наблюдений.
    # ВАЖНО: obs_keys в текущей архитектуре используется как "какие ключи нормализовать" (а не "какие выдавать").
    obs_norms: List[ObservationNorm] = []
    if cfg.normalize_obs:
        norm_keys: List[NestedKey] = list(cfg.obs_keys) if cfg.obs_keys is not None else _infer_float_obs_keys(
            base_env.observation_spec
        )

        # Самый надёжный вариант (и проще для loc/scale): по одному ObservationNorm на ключ.
        for k in norm_keys:
            tr = ObservationNorm(in_keys=[k])
            transforms.append(tr)
            obs_norms.append(tr)

    # Time-limit => truncation
    transforms.append(StepCounter(max_steps=cfg.max_steps) if cfg.max_steps is not None else StepCounter())

    env = TransformedEnv(base_env, Compose(*transforms))

    # Инициализация статистик нормализации — только после того, как трансформы прикреплены к env
    if cfg.normalize_obs and obs_norms:
        # Логика:
        # 1) Если хотя бы один из loc/scale не задан — пытаемся оценить статистики роллаутом, если init_norm_iter > 0
        # 2) Если init_norm_iter == 0 и чего-то не хватает — ставим дефолты (loc=0, scale=1) и затем применяем override
        need_stats = (cfg.loc is None) or (cfg.scale is None)

        if need_stats and cfg.init_norm_iter > 0:
            for tr in obs_norms:
                tr.init_stats(num_iter=cfg.init_norm_iter, reduce_dim=0, cat_dim=0)
        elif need_stats and cfg.init_norm_iter == 0:
            # fail-safe: без статистик задаём identity-нормализацию
            for tr in obs_norms:
                tr.loc = torch.as_tensor(0.0, device=device, dtype=torch.float32)
                tr.scale = torch.as_tensor(1.0, device=device, dtype=torch.float32)

        # Override loc/scale, если заданы пользователем.
        # Поддерживаем:
        #  - скаляр/массив -> применяется ко всем ключам (broadcast допустим)
        #  - dict {key: value} -> применяется точечно
        if cfg.loc is not None:
            if isinstance(cfg.loc, dict):
                for tr in obs_norms:
                    k = tr.in_keys[0]
                    if k in cfg.loc:
                        tr.loc = torch.as_tensor(cfg.loc[k], device=device, dtype=torch.float32)
            else:
                val = torch.as_tensor(cfg.loc, device=device, dtype=torch.float32)
                for tr in obs_norms:
                    tr.loc = val

        if cfg.scale is not None:
            if isinstance(cfg.scale, dict):
                for tr in obs_norms:
                    k = tr.in_keys[0]
                    if k in cfg.scale:
                        tr.scale = torch.as_tensor(cfg.scale[k], device=device, dtype=torch.float32)
            else:
                val = torch.as_tensor(cfg.scale, device=device, dtype=torch.float32)
                for tr in obs_norms:
                    tr.scale = val

    return env


def _format_frame(frame) -> Optional[np.ndarray]:
    """Приводим кадр к формату, удобному для imshow."""
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    # (C, H, W) -> (H, W, C)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[0] < frame.shape[-1]:
        frame = np.transpose(frame, (1, 2, 0))

    # RGBA -> RGB
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]

    # (H, W, 1) -> (H, W)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]

    return frame


def _get_actor_device(actor: torch.nn.Module) -> torch.device:
    try:
        return next(actor.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def play(
    actor: torch.nn.Module,
    env: TransformedEnv,
    *,
    fps: Optional[int] = 30,
    exp_type: Optional[ExplorationType] = None,
    figsize: Tuple[int, int] = (6, 4),
    return_frames: bool = False,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Прокатать один эпизод в готовой среде (rgb_array/human), показать кадры и вернуть статистику.

    Совместимо с TorchRL-актером, который кладёт действие в ключ "action".

    Возвращает:
      {"steps": int, "return": float, "terminated": bool, "truncated": bool, "frames": Optional[List[np.ndarray]]}
    """
    assert hasattr(env, "render"), "Окружение не поддерживает render(). Укажи корректный render_mode при создании."

    # ExplorationType по умолчанию
    if exp_type is None:
        exp_type = ExplorationType.MODE if is_acts_discrete(env.action_spec) else ExplorationType.DETERMINISTIC

    actor.eval()
    actor_device = _get_actor_device(actor)
    env_device = getattr(env, "device", torch.device("cpu"))

    # inline-рендер для Jupyter: только для rgb_array
    base = getattr(env, "base_env", env)
    inline = getattr(base, "render_mode", None) == "rgb_array"

    fig = ax = im = display_handle = None
    if inline:
        # Lazy import, чтобы модуль не требовал IPython вне ноутбука
        from IPython.display import display  # type: ignore

        fig, ax = plt.subplots(figsize=figsize)

    frames: Optional[List[np.ndarray]] = [] if return_frames else None

    td = env.reset()
    done = False
    steps = 0
    ep_return = 0.0
    last_term = False
    last_trunc = False

    try:
        while not done:
            with torch.no_grad(), set_exploration_type(exp_type):
                td_actor = td.to(actor_device)
                td_actor = actor(td_actor)
                td = td_actor.to(env_device)

            td = env.step(td)

            # Рендеринг
            if inline:
                frame = _format_frame(env.render())
                if frame is not None:
                    if im is None:
                        im = ax.imshow(frame)  # type: ignore[union-attr]
                        ax.set_axis_off()      # type: ignore[union-attr]
                        display_handle = display(fig, display_id=True)  # type: ignore[misc]
                    else:
                        im.set_data(frame)  # type: ignore[union-attr]
                        display_handle.update(fig)  # type: ignore[union-attr]
                    if frames is not None:
                        frames.append(frame.copy())
            else:
                env.render()

            term = td.get(("next", "terminated"), torch.zeros((), dtype=torch.bool))
            trunc = td.get(("next", "truncated"), torch.zeros((), dtype=torch.bool))

            last_term = bool(term.any().item()) if term.numel() > 0 else bool(term.item())
            last_trunc = bool(trunc.any().item()) if trunc.numel() > 0 else bool(trunc.item())
            done = last_term or last_trunc

            rew = td.get(("next", "reward"), None)
            if rew is not None:
                ep_return += float(rew.sum().item())

            td = step_mdp(td)

            steps += 1
            if max_steps is not None and steps >= max_steps:
                done = True

            if fps and fps > 0:
                time.sleep(max(0.0, 1.0 / float(fps)))

    finally:
        if inline and fig is not None:
            plt.close(fig)

    out: Dict[str, Any] = {
        "steps": steps,
        "return": ep_return,
        "terminated": last_term,
        "truncated": last_trunc,
    }
    if frames is not None:
        out["frames"] = frames
    return out
