from __future__ import annotations
from pathlib import Path
import torch


def save_checkpoint(policy, path: str | Path, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": policy.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_checkpoint(policy, path: str | Path, map_location=None):
    payload = torch.load(path, map_location=map_location)
    policy.load_state_dict(payload["state_dict"])
    return payload.get("metadata", {})

