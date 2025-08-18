# src/agentslab/envs/gym.py

from __future__ import annotations
from typing import Optional
from torchrl.envs import GymEnv, TransformedEnv, StepCounter
from torchrl.record import VideoRecorder

def make_gym_env(
    env_id: str = "CartPole-v1",
    seed: int = 0,
    record_video: bool = False,
    video_dir: Optional[str] = None,
    max_steps: Optional[int] = None,
    from_pixels: bool = False,
    pixels_only: bool = False,
    render_mode: Optional[str] = None,
):
    # 1) Не передаём seed в конструктор GymEnv!
    env = GymEnv(
        env_id,
        from_pixels=from_pixels,
        pixels_only=pixels_only,
        render_mode=render_mode,
    )

    transforms = []
    if max_steps is not None:
        transforms.append(StepCounter(max_steps=max_steps))
    if record_video and video_dir is not None:
        transforms.append(VideoRecorder(video_dir=video_dir, tag="rollout"))
    if transforms:
        env = TransformedEnv(env, *transforms)

    # 2) Ставим сид корректно
    env.set_seed(seed)
    return env
