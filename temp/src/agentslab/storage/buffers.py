from __future__ import annotations
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

def make_replay_buffer(size: int, batch_size: int, prefetch: int = 1, pin_memory: bool = True):
    storage = LazyMemmapStorage(size)
    rb = TensorDictReplayBuffer(
        storage=storage,
        batch_size=batch_size,
        prefetch=prefetch,
        pin_memory=pin_memory,
    )
    return rb
