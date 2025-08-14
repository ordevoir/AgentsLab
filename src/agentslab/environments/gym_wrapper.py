from __future__ import annotations
from dataclasses import dataclass
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


@dataclass
class EnvConfig:
    id: str
    render_mode: str | None = None
    record_video: bool = False
    video_dir: str | None = None
    record_stats: bool = True


def make_env(cfg: EnvConfig):
    env = gym.make(cfg.id, render_mode=cfg.render_mode)
    if cfg.record_stats:
        env = RecordEpisodeStatistics(env)
    if cfg.record_video:
        # Требуется render_mode="rgb_array" для записи видео
        assert cfg.render_mode == "rgb_array", "Set render_mode='rgb_array' to use RecordVideo"
        env = RecordVideo(env, video_folder=str(cfg.video_dir))
    return env