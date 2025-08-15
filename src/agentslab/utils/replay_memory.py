from collections import deque, namedtuple
import random

Transition = namedtuple("Transition","state action next_state reward")
class ReplayMemory:
    def __init__(self, capacity:int): self._mem = deque([], maxlen=capacity)
    def push(self, *args): self._mem.append(Transition(*args))
    def sample(self, batch_size:int): return random.sample(self._mem, batch_size)
    def __len__(self): return len(self._mem)
    def clear(self): self._mem.clear()
