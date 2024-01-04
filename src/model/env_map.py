import numpy as np
from copy import copy



EMPTY_CHAR  = ' '
WALL_CHAR   = '#'
APPLE_CHAR  = 'o'
BEAM_CHAR   = '='


# Map layout used in environment
class EnvMap:
    def __init__(
        self,
        layout,
        starting_zones
    ):
        self.layout = layout
        self.starting_zones = starting_zones


    @property
    def shape(self):
        return (
            len(self.layout[0]),
            len(self.layout)
        )


    def sample_starting_positions(self, n_positions):
        starting_positions = np.concatenate([
            np.transpose(
                np.mgrid[sz[0][0]:sz[1][0], sz[0][1]:sz[1][1]],
                (1, 2, 0)
            ).reshape(-1, 2)
            for sz in self.starting_zones
        ], axis=0)
        starting_idx = np.random.choice(starting_positions.shape[0], size=n_positions, replace=False)
        return starting_positions[starting_idx]


    def initialize(self, agents_char):
        layout = copy(self.layout)
        n_agents = len(agents_char)
        positions = self.sample_starting_positions(n_agents)

        for agent_char, position in zip(agents_char, positions):
            y, x = position
            layout[y] = layout[y][:x] + agent_char + layout[y][x + 1:]

        return layout