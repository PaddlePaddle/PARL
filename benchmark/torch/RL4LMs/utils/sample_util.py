from collections import deque
from typing import Any, List
import numpy as np


class PrioritySampler:
    def __init__(self, max_size: int = None, priority_scale: float = 0.0):
        """
        Creates a priority sampler

        Args:
            max_size (int): maximum size of the queue
            priority_scale (float): 0.0 is a pure uniform sampling, 1.0 is completely priority sampling
        """
        self.max_size = max_size
        self.items = deque(maxlen=self.max_size)
        self.item_priorities = deque(maxlen=self.max_size)
        self.priority_scale = priority_scale

    def add(self, item: Any, priority: float):
        self.items.append(item)
        self.item_priorities.append(priority)

    def sample(self, size: int) -> List[Any]:
        min_sample_size = min(len(self.items), size)
        scaled_item_priorities = np.array(
            self.item_priorities) ** self.priority_scale
        sample_probs = scaled_item_priorities / np.sum(scaled_item_priorities)
        samples = np.random.choice(
            a=self.items, p=sample_probs, size=min_sample_size)
        return samples

    def update(self, item: Any, priority: float):
        index = self.items.index(item)
        del self.items[index]
        del self.item_priorities[index]
        self.add(item, priority)

    def get_all_samples(self) -> List[Any]:
        return self.items
