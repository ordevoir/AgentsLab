from __future__ import annotations

import math


def epsilon_by_frame(step: int, start: float, end: float, decay_steps: int) -> float:
    """Exponential decay schedule used for DQN epsilon-greedy."""
    return end + (start - end) * math.exp(-1.0 * step / float(decay_steps))
