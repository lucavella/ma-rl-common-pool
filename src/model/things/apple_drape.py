import numpy as np
from scipy.signal import convolve2d
from pycolab.things import Drape
from functools import cached_property
import model



# Documentation: https://github.com/google-deepmind/pycolab/blob/6504c8065dd5322438f23a2a9026649904f3322a/pycolab/things.py#L161
class AppleDrape(Drape):
    def __init__(self, curtain, character, agents, agents_char, respawn_radius, respawn_probabilities):
        super().__init__(curtain, character)
        self.agents = agents
        self.agents_char = agents_char
        self.spawn_points = np.copy(curtain)
        self.respawn_radius = respawn_radius
        self.respawn_probabilities = respawn_probabilities


    @cached_property
    def count_filter(self):
        center = int(self.respawn_radius)
        size = center * 2 + 1
        positions = np.arange(size) - size // 2
        x, y = np.meshgrid(positions, positions)
        distances = np.sqrt(x**2 + y**2)
        distances[center, center] = 0
        return (distances <= self.respawn_radius).astype(int)


    @property
    def any_left(self):
        return np.any(self.curtain)


    def get_spawn_probabilities(self, counts, spawn_mask):
        result = np.zeros_like(spawn_mask, dtype=float)
        p_one_two, p_three_four, p_five_more = self.respawn_probabilities
        result[spawn_mask & (counts >= 1) & (counts <= 2)] = p_one_two
        result[spawn_mask & (counts >= 3) & (counts <= 4)] = p_three_four
        result[spawn_mask & (counts >= 5)] = p_five_more
        return result

    
    def update(self, actions, board, layers, backdrop, things, the_plot):
        rewards = dict()
        spawn_points = self.spawn_points & ~self.curtain

        for agent_id, agent_char in zip(self.agents, self.agents_char):
            agent = things[agent_char]
            y, x = agent.position
            rewards[agent_id] = int(self.curtain[y, x])
            self.curtain[y, x] = False
            spawn_points[y, x] = False
        the_plot.add_reward(rewards)

        apples_around = convolve2d(self.curtain, self.count_filter, mode='same', fillvalue=False)
        spawn_propabilities = self.get_spawn_probabilities(apples_around, spawn_points)
        random_values = np.random.rand(*spawn_points.shape)
        self.curtain[...] |= random_values < spawn_propabilities
