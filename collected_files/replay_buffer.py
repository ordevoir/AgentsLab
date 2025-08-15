
from __future__ import annotations
from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque
import random
import numpy as np
import torch

@dataclass
class Transition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, *args) -> None:
        self._buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self._buffer, batch_size)
        states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32)
        actions = torch.as_tensor([b.action for b in batch], dtype=torch.long).unsqueeze(-1)
        next_states = torch.as_tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32).unsqueeze(-1)
        dones = torch.as_tensor([b.done for b in batch], dtype=torch.float32).unsqueeze(-1)
        return states, actions, next_states, rewards, dones
