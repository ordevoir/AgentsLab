from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from hydra.utils import get_original_cwd


@dataclass
class CheckpointPaths:
    run_dir: Path
    algo: str
    env_id: str

    def algo_dir(self) -> Path:
        p = self.run_dir / "checkpoints" / "rl" / self.algo / self.env_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def global_dir(self) -> Path:
        root = Path(get_original_cwd())
        stamp = self.run_dir.name  # HH-MM-SS
        date = self.run_dir.parent.name  # YYYY-MM-DD
        p = root / "checkpoints" / "rl" / self.algo / self.env_id / f"{date}_{stamp}"
        p.mkdir(parents=True, exist_ok=True)
        return p


def save_checkpoint(paths: CheckpointPaths, state: Dict[str, Any], tag: str) -> Path:
    cp_dir = paths.algo_dir()
    cp_path = cp_dir / f"{tag}.ckpt"
    torch.save(state, cp_path)

    if tag in {"best", "last"}:
        gdir = paths.global_dir()
        gpath = gdir / f"{tag}.ckpt"
        torch.save(state, gpath)
    return cp_path


def load_checkpoint(path: str | Path, map_location: Optional[str] = None) -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
