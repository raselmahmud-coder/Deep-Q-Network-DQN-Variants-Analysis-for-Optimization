# memory.py

import random
import collections

class Memory:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, data):
        """Add a transition to the buffer."""
        self.buffer.append(data)
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
