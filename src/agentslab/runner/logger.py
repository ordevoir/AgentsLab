from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from torchrl.record.loggers import CSVLogger, TensorboardLogger
from torchrl.record import VideoRecorder

@dataclass
class LogSetup:
    log_dir: str
    run_name: str
    csv: CSVLogger
    tb: Optional[TensorboardLogger]
    video_recorder: Optional[VideoRecorder]

def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def prepare_logging(root: str, run_name: str, with_tensorboard: bool = True, video: bool=False) -> LogSetup:
    # create unique dir: logs/run_name/<timestamp>
    ts = _timestamp()
    log_dir = os.path.join(root, run_name, ts)
    os.makedirs(log_dir, exist_ok=True)
    csv = CSVLogger(exp_name=run_name, log_dir=log_dir)
    tb = TensorboardLogger(exp_name=run_name, log_dir=log_dir) if with_tensorboard else None
    vr = VideoRecorder(log_dir, tag="eval") if video else None
    return LogSetup(log_dir=log_dir, run_name=run_name, csv=csv, tb=tb, video_recorder=vr)

def log_metrics(loggers: LogSetup, metrics: Dict[str, Any], step: int | None = None):
    # CSV always
    loggers.csv.log_dict(metrics, step)
    if loggers.tb is not None:
        loggers.tb.log_dict(metrics, step)
