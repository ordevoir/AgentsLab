from __future__ import annotations
from typing import Optional
import torch

def get_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve device, giving priority to `preferred` when available.
    Example: get_device("cuda"), get_device("mps"), get_device("cpu")
    """
    if preferred is not None:
        try:
            dev = torch.device(preferred)
            if dev.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            if dev.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS not available")
            return dev
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
