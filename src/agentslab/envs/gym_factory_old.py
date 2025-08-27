from dataclasses import dataclass, replace
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from typing import Optional, Sequence, Union, List, Dict, Any, Tuple
import torch
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import Compose, ObservationNorm, DoubleToFloat, StepCounter
from torchrl.envs.utils import check_env_specs
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
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


def _infer_float_obs_keys(observation_spec) -> List[str]:
    """Угадываем ключи наблюдений с float dtype (кроме пиксельных/кадровых).
    Игнорируем ключи, в названии которых встречаются pixel/pixels/image/rgb/frame (без учёта регистра).
    Поддерживаем float16, bfloat16, float32, float64.
    """
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
        # Неспецифичная (не-композитная) среда — по умолчанию считаем один ключ
        # и нормализуем его, если dtype выглядит как float.
        dt = getattr(observation_spec, "dtype", None)
        if dt in float_dtypes:
            return ["observation"]
        return ["observation"]
    except Exception:
        return ["observation"]


@dataclass
class GymEnvConfig:
    env_id: str = "InvertedDoublePendulum-v4"
    render_mode: Optional[str] = None
    norm_obs: bool = True
    init_norm_iter: int = 1000
    max_steps: Optional[int] = None  # если задано — усечение эпизода по длине
    device: Union[str, torch.device] = "cpu"
    seed: Optional[int] = 0

    # Доп. опции качества жизни
    obs_keys: Optional[Sequence[str]] = None  # явное перечисление ключей для нормализации
    double_to_float: bool = True              # приводить float64→float32
    check_specs_flag: bool = False            # проверить спецификации на соответствие


def make_gym_env(cfg: GymEnvConfig) -> TransformedEnv:
    base = GymEnv(cfg.env_id, device=cfg.device, render_mode=cfg.render_mode)
    if cfg.seed is not None:
        base.set_seed(cfg.seed)

    # Автоматически определяем ключи наблюдений для нормализации (если надо)
    norm_keys = None
    if cfg.norm_obs:
        norm_keys = list(cfg.obs_keys) if cfg.obs_keys is not None else _infer_float_obs_keys(base.observation_spec)

    transforms = []
    if cfg.double_to_float:
        transforms.append(DoubleToFloat())  # сначала приводим dtype
    if cfg.norm_obs:
        transforms.append(ObservationNorm(in_keys=norm_keys))
    if cfg.max_steps is not None:
        transforms.append(StepCounter(max_steps=cfg.max_steps))
    else:
        # Либо всегда добавляйте StepCounter (полезно иметь счётчик шагов)
        transforms.append(StepCounter())

    env = TransformedEnv(base, Compose(*transforms))

    # Инициализация статистик ObservationNorm
    if cfg.norm_obs and cfg.init_norm_iter and cfg.init_norm_iter > 0:
        # Находим трансформ по типу, а не по индексу
        obs_norm = next(t for t in env.transform if isinstance(t, ObservationNorm))
        obs_norm.init_stats(num_iter=cfg.init_norm_iter, reduce_dim=0)

    if cfg.check_specs_flag:
        check_env_specs(env)

    env.reset()
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

def play_episode(
    actor: torch.nn.Module,
    cfg: GymEnvConfig,
    *,
    fps: Optional[int] = 30,
    exp_type: Optional[ExplorationType] = None,
    figsize: Tuple[int, int] = (6, 4),
    return_frames: bool = False,
) -> Dict[str, Any]:
    """
    Прокатывает один эпизод, рендерит его (inline для rgb_array или окно для human),
    возвращает статистики. Если return_frames=True — возвращает список кадров (np.ndarray).

    Параметры:
      - actor: TorchRL-совместимый актёр (Probabilistic/Deterministic).
      - cfg: конфиг окружения. По умолчанию используйте render_mode="rgb_array" для inline.
      - fps: целевой FPS для паузы между кадрами; если None/<=0 — без паузы.
      - exp_type: если None — выбирается автоматически: MODE для дискретных действий, DETERMINISTIC для непрерывных.
      - figsize: размер фигуры для Matplotlib (только для rgb_array).
      - return_frames: собрать и вернуть кадры (можно потом сделать mp4).

    Возвращает dict:
      {"steps": int, "return": float, "terminated": bool, "truncated": bool, "frames": Optional[List[np.ndarray]]}
    """
    # Сделаем локальную копию cfg, чтобы не трогать исходный (например, если захотите принудительно переключать режим)
    local_cfg = replace(cfg)
    
    # Создаём среду
    env: TransformedEnv = make_gym_env(local_cfg)

    # Выбор exploration type (если не задан)
    if exp_type is None:
        exp_type = ExplorationType.MODE if is_acts_discrete(env.action_spec) else ExplorationType.DETERMINISTIC

    # Девайс актёра
    actor.eval()
    actor_device = _get_actor_device(actor)

    # Сброс
    td = env.reset()
    done = False
    step = 0
    ep_return = 0.0
    last_term = False
    last_trunc = False

    # Inline рендер?
    inline = (local_cfg.render_mode == "rgb_array")

    # Подготовка Matplotlib (только для inline)
    fig = None
    ax = None
    im = None
    display_handle = None
    if inline:
        fig, ax = plt.subplots(figsize=figsize)

    frames: List[np.ndarray] = [] if return_frames else None

    created_env = True  # на случай, если когда-то захотите принимать env снаружи
    try:
        while not done:
            # Актёр
            with torch.no_grad(), set_exploration_type(exp_type):
                td = td.to(actor_device)
                td = actor(td)
                td = td.to(env.device)

            # Шаг среды
            td = env.step(td)

            # Рендер
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
                    if return_frames:
                        frames.append(frame.copy())
            else:
                # для human-окна достаточно вызвать render() чтобы обновить кадр
                env.render()

            # Dones
            term = td.get(("next", "terminated"), torch.zeros((), dtype=torch.bool))
            trunc = td.get(("next", "truncated"), torch.zeros((), dtype=torch.bool))
            done = bool((term | trunc).item())
            last_term = bool(term.item()) if term.numel() == 1 else bool(term.any().item())
            last_trunc = bool(trunc.item()) if trunc.numel() == 1 else bool(trunc.any().item())

            # Reward
            rew = td.get(("next", "reward"), None)
            if rew is not None:
                # одиночная среда: скаляр; на всякий случай суммируем
                ep_return += float(rew.sum().item())

            # Продвигаем MDP
            td = step_mdp(td)

            # FPS
            if fps and fps > 0:
                time.sleep(1.0 / fps)

            step += 1

    finally:
        if inline and fig is not None:
            plt.close(fig)
        if created_env and env is not None:
            env.close()

    out: Dict[str, Any] = {
        "steps": step,
        "return": ep_return,
        "terminated": last_term,
        "truncated": last_trunc,
    }
    if return_frames:
        out["frames"] = frames
    return out