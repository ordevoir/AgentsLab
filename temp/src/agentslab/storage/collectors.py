from __future__ import annotations
from torchrl.collectors import SyncDataCollector

def make_sync_collector(create_env_fn, policy, frames_per_batch: int, total_frames: int,
                        init_random_frames: int, reset_at_each_iter: bool, device):
    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        init_random_frames=init_random_frames,
        reset_at_each_iter=reset_at_each_iter,
        device=device,
    )
    return collector
