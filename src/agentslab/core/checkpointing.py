# src/agentslab/core/checkpointing.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import torch


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    directory: str,
    filename: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Save an informative PyTorch checkpoint.

    Args:
        directory: Folder to save into.
        filename: File name without extension, e.g. "best" or "last".
        model: Model to save.
        optimizer: Optimizer to save (optional).
        step: Global training step (timesteps or episodes, depending on algo).
        meta: Arbitrary metadata to store (env_id, config, metrics, etc.).

    Returns:
        The full path to the saved checkpoint file.
    """
    _ensure_dir(directory)
    ckpt_path = os.path.join(directory, f"{filename}.pt")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "meta": meta or {},
    }
    torch.save(payload, ckpt_path)

    # Also write a lightweight sidecar meta.json for quick inspection
    meta_json_path = os.path.join(directory, f"{filename}.meta.json")
    try:
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump({"step": step, **(meta or {})}, f, indent=2, ensure_ascii=False)
    except Exception:
        # Non-critical
        pass
    return ckpt_path


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a checkpoint from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location or "cpu")
