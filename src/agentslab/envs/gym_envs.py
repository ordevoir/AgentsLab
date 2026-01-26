"""
Gymnasium backend (single-agent) для AgentsLab.

Модуль предоставляет:
- GymConfig: алиас конфигурации (обратная совместимость с obs_keys)
- make_gym_env: фабрика TorchRL GymEnv + transforms (dtype, obs-norm, step counter)
- play: прокат одного эпизода с рендером (rgb_array/human)
- is_acts_discrete: утилита для определения дискретности action_spec
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
)
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import Compose, DoubleToFloat, ObservationNorm, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from agentslab.envs.env_configs import GymEnvConfig, NestedKey


__all__ = [
    "GymConfig",
    "make_gym_env",
    "play",
    "is_acts_discrete",
]


# =============================================================================
# Backward-compatible config alias
# =============================================================================


@dataclass
class GymConfig(GymEnvConfig):
    """
    Алиас GymEnvConfig для обратной совместимости.

    Исторически в notebooks использовалось поле obs_keys.
    В актуальной модели конфигурации оно называется norm_obs_keys (см. env_configs.py).

    Attributes:
        obs_keys: Устаревшее имя для norm_obs_keys (какие ключи нормализовать).
    """

    obs_keys: Optional[Sequence[NestedKey]] = None

    def __post_init__(self) -> None:
        # Миграция "на лету": obs_keys -> norm_obs_keys, если новое поле не задано
        if self.norm_obs_keys is None and self.obs_keys is not None:
            self.norm_obs_keys = list(self.obs_keys)
        super().__post_init__()


# =============================================================================
# Helpers
# =============================================================================


def _resolve_device(preferred: Union[str, torch.device, None]) -> torch.device:
    """
    Резолвит устройство для вычислений.

    Поддерживает:
      - torch.device
      - "cpu", "cuda", "cuda:N", "mps"
      - "auto" / None -> cuda -> mps -> cpu
    """
    if isinstance(preferred, torch.device):
        dev = preferred
    else:
        if preferred is None or (isinstance(preferred, str) and preferred.strip().lower() == "auto"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        dev = torch.device(str(preferred))

    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU", RuntimeWarning)
        return torch.device("cpu")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        warnings.warn("MPS not available, falling back to CPU", RuntimeWarning)
        return torch.device("cpu")
    return dev


def is_acts_discrete(action_spec: Any) -> bool:
    """
    True для всех дискретных вариантов: OneHotDiscreteTensorSpec, DiscreteTensorSpec, MultiDiscreteTensorSpec.

    Примечание:
        action_spec у TorchRL часто CompositeSpec, где реальная spec лежит по ключу "action".
        Здесь ожидается, что caller передал уже нужный "leaf spec" (как в notebooks).
    """
    return isinstance(action_spec, (OneHotDiscreteTensorSpec, DiscreteTensorSpec, MultiDiscreteTensorSpec))


def _infer_float_obs_keys(observation_spec: Any) -> List[NestedKey]:
    """
    Пытается вывести ключи float-наблюдений (не пиксели) из observation_spec.

    Для CompositeSpec возвращаем список ключей NestedKey.
    Для простого spec возвращаем ["observation"].
    """
    deny_substrings = ("pixel", "pixels", "image", "images", "rgb", "frame", "frames")
    float_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}

    try:
        # CompositeSpec-подобный
        if hasattr(observation_spec, "keys"):
            keys: List[NestedKey] = []

            # keys() бывает с разной сигнатурой в разных версиях — делаем мягко
            try:
                it = list(observation_spec.keys(True, True))
            except Exception:
                it = list(observation_spec.keys())

            for k in it:
                try:
                    sub = observation_spec[k]
                    dt = getattr(sub, "dtype", None)
                except Exception:
                    continue

                if dt in float_dtypes:
                    name = str(k).lower()
                    if any(s in name for s in deny_substrings):
                        continue
                    keys.append(k)

            return keys or ["observation"]

        # Простая spec
        dt = getattr(observation_spec, "dtype", None)
        if dt in float_dtypes:
            return ["observation"]
        return ["observation"]
    except Exception:
        return ["observation"]


def _as_tensor(x: Any, *, device: torch.device) -> torch.Tensor:
    """Приведение scalar/array/tensor к torch.Tensor на нужном устройстве."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device)
    return torch.as_tensor(x, device=device)


def _get_obs_norm_transforms(env: TransformedEnv) -> List[ObservationNorm]:
    """Достаёт все ObservationNorm трансформы из TransformedEnv (не полагаясь на индексы)."""
    tr = getattr(env, "transform", None)
    if tr is None:
        return []
    seq = getattr(tr, "transforms", None)
    if seq is None:
        return []
    return [t for t in seq if isinstance(t, ObservationNorm)]


# =============================================================================
# Factory
# =============================================================================


def make_gym_env(cfg: GymEnvConfig) -> TransformedEnv:
    """
    Создаёт TorchRL GymEnv и оборачивает трансформами согласно GymEnvConfig.

    Соответствие GymEnvConfig:
      - env_id -> GymEnv(env_name=...)
      - render_mode -> GymEnv(render_mode=...)
      - env_kwargs -> пробрасывается в GymEnv / gymnasium.make
      - seed -> base_env.set_seed
      - double_to_float -> DoubleToFloat
      - normalize_obs/init_norm_iter/loc/scale/norm_obs_keys -> ObservationNorm (+ init_stats / ручные loc/scale)
      - max_steps -> StepCounter(max_steps=...) (truncation)
    """
    device = _resolve_device(cfg.device)

    env_kwargs: Dict[str, Any] = dict(getattr(cfg, "env_kwargs", {}) or {})
    # Не допускаем конфликтов по именам с явными аргументами.
    env_kwargs.pop("render_mode", None)
    env_kwargs.pop("device", None)
    env_kwargs.pop("env_name", None)

    # Явно фиксируем backend = gymnasium
    with set_gym_backend("gymnasium"):
        # В разных версиях TorchRL kwargs мог приниматься по-разному — делаем безопасно.
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

    if cfg.seed is not None:
        try:
            base_env.set_seed(int(cfg.seed))
        except Exception:
            warnings.warn("Failed to set env seed via base_env.set_seed()", RuntimeWarning)

    transforms: List[Any] = []

    # dtype: float64 -> float32
    if cfg.double_to_float:
        transforms.append(DoubleToFloat())

    # Нормализация наблюдений
    obs_norms: List[ObservationNorm] = []
    if cfg.normalize_obs:
        norm_keys = list(cfg.norm_obs_keys) if cfg.norm_obs_keys is not None else _infer_float_obs_keys(
            base_env.observation_spec
        )

        # Эталонный подход: один ObservationNorm на ключ.
        # Это проще для точечного loc/scale (dict {key: value}).
        for k in norm_keys:
            tr = ObservationNorm(in_keys=[k])
            transforms.append(tr)
            obs_norms.append(tr)

    # Time-limit => truncation
    transforms.append(StepCounter(max_steps=cfg.max_steps) if cfg.max_steps is not None else StepCounter())

    env = TransformedEnv(base_env, Compose(*transforms))

    # Инициализация статистик нормализации — только после того, как трансформы прикреплены к env
    if cfg.normalize_obs and obs_norms:
        need_stats = (cfg.loc is None) or (cfg.scale is None)

        # 1) Если loc/scale не заданы — пытаемся оценить статистики роллаутом (если init_norm_iter > 0)
        if need_stats and cfg.init_norm_iter > 0:
            for tr in obs_norms:
                try:
                    tr.init_stats(num_iter=int(cfg.init_norm_iter), reduce_dim=0, cat_dim=0)
                except Exception as err:
                    warnings.warn(f"ObservationNorm.init_stats failed: {err}", RuntimeWarning)

        # 2) Если статистик нет и init_norm_iter == 0 — fail-safe: identity-нормализация
        if need_stats and cfg.init_norm_iter == 0:
            for tr in obs_norms:
                tr.loc = torch.as_tensor(0.0, device=device, dtype=torch.float32)
                tr.scale = torch.as_tensor(1.0, device=device, dtype=torch.float32)

        # 3) Override loc/scale, если заданы пользователем.
        # Поддерживаем:
        #  - scalar/array -> применяется ко всем ключам (broadcast допустим)
        #  - dict {key: value} -> применяется точечно по ключу
        if cfg.loc is not None:
            if isinstance(cfg.loc, dict):
                for tr in obs_norms:
                    k = tr.in_keys[0]
                    if k in cfg.loc:
                        tr.loc = _as_tensor(cfg.loc[k], device=device).to(dtype=torch.float32)
            else:
                val = _as_tensor(cfg.loc, device=device).to(dtype=torch.float32)
                for tr in obs_norms:
                    tr.loc = val

        if cfg.scale is not None:
            if isinstance(cfg.scale, dict):
                for tr in obs_norms:
                    k = tr.in_keys[0]
                    if k in cfg.scale:
                        tr.scale = _as_tensor(cfg.scale[k], device=device).to(dtype=torch.float32)
            else:
                val = _as_tensor(cfg.scale, device=device).to(dtype=torch.float32)
                for tr in obs_norms:
                    tr.scale = val

    return env


# =============================================================================
# Playback / rendering helper
# =============================================================================


def _format_frame(frame: Any) -> Optional[Any]:
    """
    Приводит кадр к формату, удобному для imshow.

    Возвращает numpy.ndarray (если numpy доступен) или torch.Tensor, если numpy не импортирован.
    """
    if frame is None:
        return None

    np = sys.modules.get("numpy")
    if isinstance(frame, torch.Tensor):
        if np is not None:
            frame = frame.detach().cpu().numpy()
        else:
            frame = frame.detach().cpu()

    # Если numpy доступен, применяем более удобные преобразования
    if np is not None and isinstance(frame, np.ndarray):
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
      {"steps": int, "return": float, "terminated": bool, "truncated": bool, "frames": Optional[List[Any]]}

    Notes:
        - Для inline-рендера требуется render_mode="rgb_array" и наличие matplotlib/numpy.
        - done-сигнал читается по ("next","terminated") и ("next","truncated").
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
    frames: Optional[List[Any]] = [] if return_frames else None

    if inline:
        # lazy imports (как в env_configs.py — не требуем зависимости, пока не нужно)
        import matplotlib.pyplot as plt  # noqa: F401
        from IPython.display import display  # type: ignore

        import matplotlib.pyplot as plt  # type: ignore[no-redef]
        fig, ax = plt.subplots(figsize=figsize)  # type: ignore[assignment]

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
                        # если это numpy, copy() безопаснее; если tensor — тоже ок через clone()
                        try:
                            frames.append(frame.copy())  # type: ignore[attr-defined]
                        except Exception:
                            try:
                                frames.append(frame.clone())
                            except Exception:
                                frames.append(frame)
            else:
                env.render()

            term = td.get(("next", "terminated"), torch.zeros((), dtype=torch.bool, device=td.device))
            trunc = td.get(("next", "truncated"), torch.zeros((), dtype=torch.bool, device=td.device))

            last_term = bool(term.any().item()) if term.numel() > 0 else bool(term.item())
            last_trunc = bool(trunc.any().item()) if trunc.numel() > 0 else bool(trunc.item())
            done = last_term or last_trunc

            rew = td.get(("next", "reward"), None)
            if rew is not None:
                ep_return += float(rew.sum().item())

            td = step_mdp(td)

            steps += 1
            if max_steps is not None and steps >= int(max_steps):
                done = True

            if fps and fps > 0:
                time.sleep(max(0.0, 1.0 / float(fps)))

    finally:
        if inline and fig is not None:
            import matplotlib.pyplot as plt  # lazy
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
