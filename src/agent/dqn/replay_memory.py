import numpy as np
import random
from collections import namedtuple, deque



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayMemory(object):

    def __init__(self, capacity, minibatch_size=32):
        self.memory = deque([], maxlen=capacity)
        self.index = 0
        self.minibatch_size = int(minibatch_size)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        self.index = (self.index + 1)

    def sample(self, minibatch_size):
        return random.sample(self.memory, minibatch_size)

    def __len__(self):
        return len(self.memory)

    @property 
    def current_index(self):
        return self.index


    ## dit hier doet eigenlijk hetzelfde als sample ongeveer 
    def get_minibatch_indices(self):
            indices = []
            while len(indices) < self.minibatch_size:
                index = np.random.randint(low=0, high=self.current_index)
                indices.append(index)
            return indices

    def generate_minibatch_samples(self, indices):
        state_batch, action_batch, next_state_batch, reward_batch = [], [], [], []
    
        for index in indices:
            selected_mem = self.memory[index]
            state_batch.append(selected_mem.state)
            action_batch.append(selected_mem.action)
            next_state_batch.append(selected_mem.next_state)
            reward_batch.append(selected_mem.reward, dtype=torch.float32)
            
        return (
            torch.stack(state_batch, dim=0),
            torch.stack(action_batch, dim=0),
            torch.stack(next_state_batch, dim=0),
            torch.stack(reward_batch, dim=0)
        )