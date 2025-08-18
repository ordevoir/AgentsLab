from dataclasses import dataclass
from typing import Optional, Union
import torch
from torchrl.collectors import SyncDataCollector

@dataclass
class CollectorConfig:
    frames_per_batch: int = 2048
    total_frames: int = 1_000_000
    split_trajs: bool = False
    device: Union[str, torch.device] = "cpu"

def build_sync_collector(cfg: CollectorConfig, env, policy):
    return SyncDataCollector(
        env,
        policy,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        split_trajs=cfg.split_trajs,
        device=cfg.device,
    )
