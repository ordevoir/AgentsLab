from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Deque

Transition = namedtuple("Transition", "state action next_state reward")


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self._memory: Deque = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        self._memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self._memory, batch_size)

    def __len__(self) -> int:  # pragma: no cover
        return len(self._memory)

    def clear(self) -> None:  # pragma: no cover
        self._memory.clear()
