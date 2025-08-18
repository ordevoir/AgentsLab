from dataclasses import dataclass
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement

@dataclass
class BufferConfig:
    frames_per_batch: int

def build_onpolicy_buffer(cfg: BufferConfig) -> ReplayBuffer:
    return ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.frames_per_batch),
        sampler=SamplerWithoutReplacement()
    )
