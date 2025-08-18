from __future__ import annotations
from torchrl.collectors import SyncDataCollector

def make_sync_collector(create_env_fn, policy, frames_per_batch: int, total_frames: int, device=None, reset_at_each_iter=False):
    return SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        reset_at_each_iter=reset_at_each_iter,
    )
