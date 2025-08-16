# src/agentslab/core/metrics.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable


@dataclass
class MovingAverage:
    window_size: int
    values: Deque[float] = field(default_factory=deque)

    def update(self, value: float) -> float:
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.popleft()
        return self.value

    @property
    def value(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def extend(self, iterable: Iterable[float]) -> float:
        for v in iterable:
            self.update(float(v))
        return self.value
