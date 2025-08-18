from __future__ import annotations
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage as RB_LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

def make_replay_buffer_dqn(max_size: int = 50_000, batch_size: int = 32, device=None):
    return TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=max_size, device=device), batch_size=batch_size)

def make_replay_buffer_ppo(storage_size: int, batch_size: int):
    return ReplayBuffer(storage=RB_LazyTensorStorage(storage_size), sampler=SamplerWithoutReplacement(), batch_size=batch_size)
