from dataclasses import dataclass
from typing import Union, Optional, Sequence, List, Dict, Any, Tuple
from IPython.display import display
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import Compose, ObservationNorm, DoubleToFloat, StepCounter
from torchrl.data.tensor_specs import (
    OneHotDiscreteTensorSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
)

def is_acts_discrete(action_spec) -> bool:
    """True для всех дискретных вариантов сред:
    OneHotDiscreteTensorSpec, DiscreteTensorSpec, MultiDiscreteTensorSpec"""
    return isinstance(
        action_spec,
        (OneHotDiscreteTensorSpec, DiscreteTensorSpec, MultiDiscreteTensorSpec),
    )

@dataclass
class GymConfig:
    env_id: str
    render_mode: Optional[str] = None
    device: Union[str, torch.device] = "cpu"
    seed: Optional[int] = None
    max_steps: Optional[int] = None
    normalize_obs: bool = True
    init_norm_iter: int = 1000
    obs_keys: Optional[Sequence[str]] = None
    # loc/scale можно передавать как число, список или тензор
    loc: Optional[Union[float, Sequence[float], torch.Tensor]] = None
    scale: Optional[Union[float, Sequence[float], torch.Tensor]] = None
    double_to_float: bool = True


def _infer_float_obs_keys(observation_spec) -> List[str]:
    """Выбираем float-наблюдения (не пиксели)."""
    deny_substrings = ("pixel", "pixels", "image", "images", "rgb", "frame", "frames")
    float_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}

    try:
        if hasattr(observation_spec, "keys"):
            keys = []
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


def make_gym_env(cfg: GymConfig) -> TransformedEnv:

    base_env = GymEnv(env_name=cfg.env_id, 
                      device=cfg.device,
                      render_mode=cfg.render_mode
                      )

    if cfg.seed is not None:
        base_env.set_seed(cfg.seed)  # корректный способ сидировать TorchRL-окружение
    # https://docs.pytorch.org/rl/...: env.set_seed(...) существует и возвращает следующий сид

    transforms = []

    # Сразу приводим double->float (как в туториалах TorchRL).
    if cfg.double_to_float:
        transforms.append(DoubleToFloat())

    # Нормализация наблюдений (обычно одного ключа "observation").
    obs_norm = None
    if cfg.normalize_obs:
        norm_keys = list(cfg.obs_keys) if cfg.obs_keys is not None else _infer_float_obs_keys(base_env.observation_spec)
        # Берём один «лучший» ключ (если нужно несколько — создайте по ObservationNorm на ключ)
        selected_key = norm_keys[0]
        obs_norm = ObservationNorm(in_keys=[selected_key])
        transforms.append(obs_norm)

    # Счётчик шагов
    if cfg.max_steps is not None:
        transforms.append(StepCounter(max_steps=cfg.max_steps))
    else:
        transforms.append(StepCounter())

    env = TransformedEnv(base_env, Compose(*transforms))

    # Инициализация статистик НОРМАЛИЗАЦИИ — только после того как трансформер прикреплён к env!
    if cfg.normalize_obs and obs_norm is not None:
        # Если лок/скейл не заданы — оценим их случайным роллаутом (канонически reduce_dim=0, cat_dim=0)
        if cfg.loc is None or cfg.scale is None:
            # можно явно указать key=selected_key; по умолчанию берётся первый in_key
            obs_norm.init_stats(num_iter=cfg.init_norm_iter, reduce_dim=0, cat_dim=0)
        # Если переданы loc/scale — выставляем явно
        if cfg.loc is not None:
            obs_norm.loc = torch.as_tensor(cfg.loc, device=cfg.device, dtype=torch.float32)
        if cfg.scale is not None:
            obs_norm.scale = torch.as_tensor(cfg.scale, device=cfg.device, dtype=torch.float32)

    return env


def _format_frame(frame) -> Optional[np.ndarray]:
    """Приводим кадр к формату, удобному для imshow."""
    
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    # Если каналы идут первым измерением (C, H, W) и C<width
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[0] < frame.shape[-1]:
        frame = np.transpose(frame, (1, 2, 0))

    # RGBA -> RGB
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]

    # Одноканальные кадры -> (H, W)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]

    return frame


def _get_actor_device(actor) -> torch.device:
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

    Параметры:
      - actor: torch.nn.Module (TorchRL policy / Probabilistic/Deterministic).
      - env: уже созданный TransformedEnv (например, с render_mode="rgb_array").
      - fps: целевой FPS для паузы между кадрами (если None/<=0 — без паузы).
      - exp_type: ExplorationType; если None — MODE для дискретных действий, иначе DETERMINISTIC.
      - figsize: размер фигуры для Matplotlib при rgb_array.
      - return_frames: собрать кадры и вернуть в выходном dict.
      - max_steps: жёсткий лимит шагов (если нужно).

    Возвращает:
      {"steps": int, "return": float, "terminated": bool, "truncated": bool, "frames": Optional[List[np.ndarray]]}
    """
    assert hasattr(env, "render"), "Окружение не поддерживает render(). Укажи корректный render_mode при создании."

    # Выбор exploration type по умолчанию
    # Выбор exploration type (если не задан)
    if exp_type is None:
        exp_type = ExplorationType.MODE if is_acts_discrete(env.action_spec) else ExplorationType.DETERMINISTIC

    # Девайсы
    actor.eval()
    actor_device = _get_actor_device(actor)
    env_device = env.device if hasattr(env, "device") else torch.device("cpu")

    # Признак inline-рендера
    inline = getattr(getattr(env, "base_env", env), "render_mode", None) == "rgb_array"

    # Подготовка визуализации
    fig = ax = im = display_handle = None
    if inline:
        fig, ax = plt.subplots(figsize=figsize)

    frames: Optional[List[np.ndarray]] = [] if return_frames else None

    # Инициализация эпизода
    td = env.reset()
    done = False
    steps = 0
    ep_return = 0.0
    last_term = False
    last_trunc = False

    try:
        while not done:
            # Подаём td в актёр -> получаем действие в ключе "action"
            with torch.no_grad(), set_exploration_type(exp_type):
                td_actor = td.to(actor_device)
                td_actor = actor(td_actor)
                td = td_actor.to(env_device)

            # Шаг среды
            td = env.step(td)

            # Рендеринг
            if inline:
                frame = _format_frame(env.render())
                if frame is not None:
                    if im is None:
                        im = ax.imshow(frame)
                        ax.set_axis_off()
                        display_handle = display(fig, display_id=True)
                    else:
                        im.set_data(frame)
                        display_handle.update(fig)
                    if frames is not None:
                        frames.append(frame.copy())
            else:
                env.render()  # для "human" просто обновляем окно

            # Dones
            term = td.get(("next", "terminated"), torch.zeros((), dtype=torch.bool))
            trunc = td.get(("next", "truncated"), torch.zeros((), dtype=torch.bool))

            # Булевы флаги (учёт возможных батчевых форм)
            last_term = bool(term.any().item()) if term.numel() > 0 else bool(term.item())
            last_trunc = bool(trunc.any().item()) if trunc.numel() > 0 else bool(trunc.item())
            done = last_term or last_trunc

            # Награда
            rew = td.get(("next", "reward"), None)
            if rew is not None:
                ep_return += float(rew.sum().item())

            # Шифтируем ("next" -> root) для следующего шага
            td = step_mdp(td)

            # Лимит по шагам
            steps += 1
            if max_steps is not None and steps >= max_steps:
                done = True

            # Регулировка FPS
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
