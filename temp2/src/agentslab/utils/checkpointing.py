
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import pathlib
import shutil
import json
import time

@dataclass
class Checkpoint:
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    step: int
    meta: Optional[Dict[str, Any]] = None

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, meta: Dict[str, Any] | None = None) -> None:
    path_obj = pathlib.Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = Checkpoint(model.state_dict(), optimizer.state_dict(), step, meta).__dict__
    torch.save(payload, str(path_obj))
    # also write meta.json next to the checkpoint (optional, human-readable)
    try:
        if meta is not None:
            with open(path_obj.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
    except Exception:
        pass

def copy_as_last(ckpt_file: str, last_name: str = "last.pt") -> str:
    """Copy ckpt_file to sibling 'last.pt' (works on Windows too)."""
    src = pathlib.Path(ckpt_file)
    dst = src.with_name(last_name)
    shutil.copyfile(src, dst)
    return str(dst)

def load_checkpoint(path: str) -> Checkpoint:
    data = torch.load(path, map_location="cpu")
    if "meta" not in data:
        data["meta"] = None
    return Checkpoint(**data)
