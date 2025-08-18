from __future__ import annotations
from typing import Optional, Callable

from torchrl.collectors import SyncDataCollector
from torchrl.envs import EnvBase

def make_sync_collector(
    env: EnvBase,
    policy,
    frames_per_batch: int = 256,
    init_random_frames: int = 1000,
    total_frames: int = -1,
):
    return SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        init_random_frames=init_random_frames,
    )
