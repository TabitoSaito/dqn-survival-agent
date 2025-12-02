from typing import NamedTuple, Any
from collections import deque
import random


class Experiences(NamedTuple):
    state: Any
    action: Any
    reward: Any
    next_state: Any
    done: Any
    mask: Any
    next_mask: Any


class ReplayMemory:
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experiences(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
