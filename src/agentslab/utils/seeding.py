# src/agentslab/utils/seeding.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Set seeds for python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic_torch
    torch.backends.cudnn.deterministic = deterministic_torch
    os.environ["PYTHONHASHSEED"] = str(seed)
