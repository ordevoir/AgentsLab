from dataclasses import dataclass, field
from typing import Optional, Literal, Any
import torch
import numpy as np

from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import RewardSum, Compose, StepCounter

# опционально: из torchrl.record для рендера
from torchrl.record import PixelRenderTransform

@dataclass
class VMASConfig:
    # Базовая конфигурация
    scenario: str = "navigation"
    continuous_actions: bool = True
    num_envs: int = 64
    device: torch.device | str = "cpu"
    seed: Optional[int] = None

    # Лимит шагов
    max_steps: Optional[int] = 100
    # True -> StepCounter выставит 'truncated' по лимиту шагов
    # False -> передадим max_steps в VmasEnv и получим 'terminated' по лимиту
    truncate_on_max_steps: bool = True

    # Агрегация вознаграждения за эпизод
    sum_reward: bool = True

    # Сценарно-специфичные параметры (попадают в конструктор VMAS-сценария)
    # Пример: {"n_agents": 3, "lidar_range": 5.0, ...}
    scenario_kwargs: dict[str, Any] = field(default_factory=dict)

    # Рендер
    render_mode: Optional[Literal["rgb_array", "human"]] = None
    pixels_key: str = "pixels"
    # Дополнительные аргументы прямо в env.render(...), если нужны
    render_kwargs: dict[str, Any] = field(default_factory=dict)

def make_vmas_env(cfg: VMASConfig):
    # Семантика лимита шагов: либо StepCounter->truncated, либо termination в самой среде
    vmas_max_steps = None if (cfg.truncate_on_max_steps and cfg.max_steps is not None) else cfg.max_steps

    env = VmasEnv(
        scenario=cfg.scenario,
        num_envs=cfg.num_envs,
        continuous_actions=cfg.continuous_actions,
        max_steps=vmas_max_steps,
        device=cfg.device,
        seed=cfg.seed,
        # kwargs прокидываются в сценарий VMAS:
        **cfg.scenario_kwargs,
    )

    transforms = []

    if cfg.truncate_on_max_steps and cfg.max_steps is not None:
        transforms.append(StepCounter(max_steps=cfg.max_steps))

    if cfg.sum_reward:
        # VMAS в TorchRL по умолчанию группирует агентов в "agents"
        transforms.append(
            RewardSum(
                in_keys=[("agents", "reward")],
                out_keys=[("agents", "episode_reward")],
            )
        )

    if cfg.render_mode is not None:
        # Узнаем фактический размер батча у среды (надежнее, чем cfg.num_envs)
        B = env.batch_size[0] if len(env.batch_size) else 1

        def _pixels_preproc(x, B=B):
            """
            Приводим результат render(...) к форме [B, H, W, C]:
            - если список кадров длины B -> np.stack по оси 0
            - если один кадр HWC -> «расширяем» на B (дублируем для всей батчи)
            - делаем массив континуальным в памяти (важно для Matplotlib/FFMPEG)
            """
            if isinstance(x, (list, tuple)):
                # список кадров по всем параллельным энвам
                x = np.stack([np.ascontiguousarray(f) for f in x], axis=0)
            else:
                x = np.ascontiguousarray(x)
                if x.ndim == 3:  # HWC -> BHWC
                    x = np.broadcast_to(x, (B, *x.shape)).copy()
                # если уже BHWC, ничего не делаем
            return x

        transforms.append(
            PixelRenderTransform(
                out_keys=[cfg.pixels_key],
                mode=cfg.render_mode,         # "rgb_array"
                preproc=_pixels_preproc,      # <-- ключ: приводим к [B,H,W,C]
                as_non_tensor=False,          # храним как torch.Tensor
                **(cfg.render_kwargs or {}),  # можно передать index=...
            )
        )

    if transforms:
        env = TransformedEnv(env, Compose(*transforms))
    return env



# from dataclasses import dataclass
# import torch
# from torchrl.envs.libs.vmas import VmasEnv
# from torchrl.envs import TransformedEnv
# from torchrl.envs.transforms import RewardSum, Compose

# @dataclass
# class VMASConfig:
#     scenario: str = "navigation"
#     n_agents: int = 3
#     continuous_actions: bool = True
#     max_steps: int = 100
#     num_envs: int = 64
#     device: torch.device | str = "cpu"
#     sum_reward: bool = True

# def make_vmas_env(cfg: VMASConfig):
#     env = VmasEnv(
#         scenario=cfg.scenario,
#         num_envs=cfg.num_envs,
#         continuous_actions=cfg.continuous_actions,
#         max_steps=cfg.max_steps,
#         device=cfg.device,
#         n_agents=cfg.n_agents,
#     )
#     transforms = []
#     if cfg.sum_reward:
#         transforms.append(RewardSum(in_keys=[("agents","reward")], out_keys=[("agents","episode_reward")]))
#     if transforms:
#         env = TransformedEnv(env, Compose(*transforms))
#     return env



