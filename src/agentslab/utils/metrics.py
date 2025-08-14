from __future__ import annotations
import numpy as np

def moving_mean(x, w):
    if len(x) == 0:
        return 0.0
    return float(np.mean(x[-w:]))