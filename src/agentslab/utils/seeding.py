from __future__ import annotations

import os
import random
from typing import Optional
import numpy as np
import torch

def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set global RNG seeds for python, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Disable cudnn benchmark for reproducibility
        torch.backends.cudnn.benchmark = False

def seed_from_env() -> int:
    """Read seed from env var AGENTSLAB_SEED if set, else generate reasonably random seed."""
    s = os.getenv("AGENTSLAB_SEED")
    if s is not None:
        try:
            return int(s)
        except Exception:
            pass
    return int.from_bytes(os.urandom(4), "little")
