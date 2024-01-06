import numpy as np
from pycolab.things import Drape
from ..actions import TAG_AHEAD, ORIENT_NORTH, ORIENT_EAST, ORIENT_SOUTH, ORIENT_WEST
from ..env_map import WALL_CHAR



# Documentation: https://github.com/google-deepmind/pycolab/blob/6504c8065dd5322438f23a2a9026649904f3322a/pycolab/things.py#L161
# Example: https://github.com/google-deepmind/pycolab/blob/6504c8065dd5322438f23a2a9026649904f3322a/pycolab/examples/aperture.py#L162
class BeamDrape(Drape):
    def __init__(self, curtain, character, agents, agents_char, beam_width, beam_range, stop_walls=True):
        super().__init__(curtain, character)
        self.agents = agents
        self.agents_char = agents_char
        self.radius = beam_width // 2
        self.range = beam_range
        self.stop_walls = stop_walls


    def beam_mask(self, orientation, position, layers):
        mask = np.zeros_like(self.curtain, dtype=bool)
        h, w = self.curtain.shape
        y, x = position
        c_range = self.radius // 2

        if orientation == ORIENT_NORTH:
            y_min = max(y - self.range, 0)
            y_max = min(y, h)
            x_min = max(x - self.radius, 0)
            x_max = min(x + self.radius + 1, w)
            if self.stop_walls:
                walls = layers[WALL_CHAR][y_min:y_max, x_min:x_max]
                c_range = x - x_min
                first = walls.shape[0] - 1
                walls[first, :c_range + 1] = np.cumsum(walls[first, c_range::-1], dtype=bool)[::-1]
                walls[first, c_range:] = np.cumsum(walls[first, c_range:], dtype=bool)
                walls_block = np.cumsum(walls[::-1], axis=0, dtype=bool)[::-1]
        if orientation == ORIENT_EAST:
            y_min = max(y - self.radius, 0)
            y_max = min(y + self.radius + 1, h)
            x_min = max(x + 1, 0)
            x_max = min(x + self.range + 1, w)
            if self.stop_walls:
                walls = layers[WALL_CHAR][y_min:y_max, x_min:x_max]
                c_range = y - y_min
                first = 0
                walls[:c_range + 1, first] = np.cumsum(walls[c_range::-1, first], dtype=bool)[::-1]
                walls[c_range:, first] = np.cumsum(walls[c_range:, first], dtype=bool)
                walls_block = np.cumsum(walls, axis=1, dtype=bool)
        if orientation == ORIENT_SOUTH:
            y_min = max(y + 1, 0)
            y_max = min(y + self.range + 1, h)
            x_min = max(x - self.radius, 0)
            x_max = min(x + self.radius + 1, w)
            if self.stop_walls:
                walls = layers[WALL_CHAR][y_min:y_max, x_min:x_max]
                c_range = x - x_min
                first = 0
                walls[first, :c_range + 1] = np.cumsum(walls[first, c_range::-1], dtype=bool)[::-1]
                walls[first, c_range:] = np.cumsum(walls[first, c_range:], dtype=bool)
                walls_block = np.cumsum(walls, axis=0, dtype=bool)
        if orientation == ORIENT_WEST:
            y_min = max(y - self.radius, 0)
            y_max = min(y + self.radius + 1, h)
            x_min = max(x - self.range, 0)
            x_max = min(x, w)
            if self.stop_walls:
                walls = layers[WALL_CHAR][y_min:y_max, x_min:x_max]
                c_range = y - y_min
                first = walls.shape[1] - 1
                walls[:c_range + 1, first] = np.cumsum(walls[c_range::-1, first], dtype=bool)[::-1]
                walls[c_range:, first] = np.cumsum(walls[c_range:, first], dtype=bool)
                walls_block = np.cumsum(walls[:, ::-1], axis=1, dtype=bool)[:, ::-1]

        if self.stop_walls:
            mask[y_min:y_max, x_min:x_max] = ~walls_block
        else:
            mask[y_min:y_max, x_min:x_max] = True
        return mask


    def update(self, actions, board, layers, backdrop, things, the_plot):
        self.curtain[...] = False
        if actions is not None:
            for agent_id, agent_char in zip(self.agents, self.agents_char):
                agent = things[agent_char]
                action = actions.get(agent_id)
                if action == TAG_AHEAD and not agent.tagged:
                    beam = self.beam_mask(
                        agent.orientation,
                        agent.position,
                        layers
                    )
                    self.curtain[...] |= beam