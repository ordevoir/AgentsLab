from __future__ import annotations
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
import pathlib
import time

class TBLogger:
    """Lightweight TensorBoard logger."""
    def __init__(self, log_dir: str) -> None:
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.start_time = time.time()

    def log(self, metrics: Dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
