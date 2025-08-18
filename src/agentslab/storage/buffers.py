from __future__ import annotations

from torchrl.data import ReplayBuffer, LazyTensorStorage, PrioritizedSampler

def make_replay_buffer(capacity: int = 100_000, prioritized: bool = False, alpha: float = 0.7, beta: float = 0.5):
    storage = LazyTensorStorage(capacity)
    if prioritized:
        sampler = PrioritizedSampler(
            max_capacity=capacity, alpha=alpha, beta=beta
        )
        rb = ReplayBuffer(storage=storage, sampler=sampler)
    else:
        rb = ReplayBuffer(storage=storage)
    return rb
