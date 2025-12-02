from collections import deque
from utils.constants import Experiences
import random


class ReplayMemory:
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experiences(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
