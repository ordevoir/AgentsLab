from dataclasses import dataclass
from typing import Optional, Sequence, Union, List

import torch
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import Compose, ObservationNorm, DoubleToFloat, StepCounter
from torchrl.envs.utils import check_env_specs


# def _infer_float_obs_keys(observation_spec) -> List[str]:
#     """Пытаемся угадать ключи наблюдений с float dtype (кроме пикселей)."""
#     try:
#         if hasattr(observation_spec, "keys"):
#             keys = []
#             for k in observation_spec.keys(True, True):
#                 sub = observation_spec[k]
#                 if getattr(sub, "dtype", None) in (torch.float32, torch.float64):
#                     # по конвенции игнорируем пиксельные ключи
#                     if "pixel" in str(k) or "image" in str(k):
#                         continue
#                     keys.append(k)
#             return keys or ["observation"]
#         # Неспецифичная среда — считаем, что ключ один
#         return ["observation"]
#     except Exception:
#         return ["observation"]

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




# from dataclasses import dataclass
# from typing import Optional, Sequence, Union, Dict, Any

# import torch
# from torchrl.envs import GymEnv, TransformedEnv
# from torchrl.envs.transforms import Compose, ObservationNorm, DoubleToFloat, StepCounter

# @dataclass
# class GymEnvConfig:
#     env_id: str = "InvertedDoublePendulum-v4"
#     render_mode: Optional[str] = None
#     norm_obs: bool = True
#     init_norm_iter: int = 1000
#     max_steps: Optional[int] = None  # if not None, counted by StepCounter
#     device: Union[str, torch.device] = "cpu"
#     seed: Optional[int] = 0

# def make_gym_env(cfg: GymEnvConfig) -> TransformedEnv:
#     base = GymEnv(cfg.env_id, device=cfg.device, render_mode=cfg.render_mode)
#     if cfg.seed is not None:
#         base.set_seed(cfg.seed)
#     transforms = []
#     if cfg.norm_obs:
#         transforms.append(ObservationNorm(in_keys=["observation"]))
#     transforms += [DoubleToFloat(), StepCounter()]
#     env = TransformedEnv(base, Compose(*transforms))
#     # init stats for ObservationNorm
#     if cfg.norm_obs and cfg.init_norm_iter and cfg.init_norm_iter > 0:
#         obs_norm = env.transform[0]  # first of Compose
#         obs_norm.init_stats(num_iter=cfg.init_norm_iter, reduce_dim=0)
#     return env
