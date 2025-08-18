from __future__ import annotations
from dataclasses import dataclass, field
import datetime
import os

def default_run_name(algo: str, env_id: str, seed: int) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_env = env_id.replace("/", "_")
    return f"{ts}_{algo}_{safe_env}_seed{seed}"

@dataclass
class PathsConfig:
    project_root: str
    logs_dir: str = "logs"
    ckpt_dir: str = "checkpoints"
    run_name: str = ""
    def ensure(self) -> None:
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
    @property
    def logs_path(self) -> str:
        return os.path.join(self.project_root, self.logs_dir, self.run_name)
    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.project_root, self.ckpt_dir, self.run_name)

@dataclass
class EnvConfig:
    env_id: str = "CartPole-v1"
    seed: int = 0
    render_mode: str | None = None

@dataclass
class ModelConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])

@dataclass
class PolicyConfig:
    eps_init: float = 1.0
    eps_end: float = 0.05
    anneal_steps: int = 100_000

@dataclass
class CollectorConfig:
    frames_per_batch: int = 256
    total_frames: int = 200_000
    init_random_frames: int = 1_000
    reset_at_each_iter: bool = False

@dataclass
class ReplayBufferConfig:
    size: int = 200_000
    batch_size: int = 256
    prefetch: int = 1
    pin_memory: bool = True

@dataclass
class LossConfig:
    gamma: float = 0.99
    double_dqn: bool = True
    delay_value: bool = True
    loss_function: str = "l2"

@dataclass
class OptimConfig:
    lr: float = 1e-3
    grad_clip_norm: float = 10.0

@dataclass
class TargetUpdateConfig:
    tau: float = 0.01

@dataclass
class TrainConfig:
    log_interval_frames: int = 5_000
    eval_interval_frames: int = 20_000
    utd_ratio: int = 1
    max_train_steps: int | None = None

@dataclass
class EvalConfig:
    episodes: int = 10
    max_steps_per_ep: int = 10_000
    deterministic: bool = True
