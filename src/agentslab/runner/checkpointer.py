from __future__ import annotations
import os, time, json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch

@dataclass
class CheckpointPaths:
    root: str
    run_name: str
    timestamp: str
    dir: str

def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def prepare_checkpoint_dir(root: str, run_name: str) -> CheckpointPaths:
    ts = _timestamp()
    d = os.path.join(root, run_name, ts)
    os.makedirs(d, exist_ok=True)
    return CheckpointPaths(root=root, run_name=run_name, timestamp=ts, dir=d)

def save_checkpoint(chk: CheckpointPaths, filename: str, state: Dict[str, Any]) -> str:
    path = os.path.join(chk.dir, filename)
    torch.save(state, path)
    # also write a small JSON index for quick inspection
    meta = {k: v for k, v in state.items() if isinstance(v, (int, float, str))}
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return path

def latest_run_dir(root: str, run_name: str) -> Optional[str]:
    base = os.path.join(root, run_name)
    if not os.path.isdir(base):
        return None
    runs = sorted(os.listdir(base))
    return os.path.join(base, runs[-1]) if runs else None
