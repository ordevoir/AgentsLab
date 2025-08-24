
from dataclasses import dataclass
from typing import Any

@dataclass
class PPOConfigs:
    actor: Any = None
    critic: Any = None
    gamma: float = 0.99
    lmbda: float = 0.95
    frames_per_batch: int = 1000
    total_frames: int = 10_000
    clip_epsilon: float  = 0.2  
    entropy_eps: float  = 1e-4
    lr: float  = 3e-4
    critic_coeff: float = 1.0
    loss_critic_type: str = "smooth_l1"
    max_grad_norm: float = 1.0
    num_epochs: int = 10
    sub_batch_size: int = 64
    eval_every: int = 5