import numpy as np
from pycolab.prefab_parts.sprites import MazeWalker
import model



# References:
# https://github.com/google-deepmind/pycolab/blob/6504c8065dd5322438f23a2a9026649904f3322a/pycolab/examples/classics/four_rooms.py
# https://github.com/google-deepmind/pycolab/blob/6504c8065dd5322438f23a2a9026649904f3322a/pycolab/prefab_parts/sprites.py#L27C13-L27C13
class AgentSprite(MazeWalker):
    def __init__(self, corner, position, character, agent_id, timeout_duration, orientation=None):
        super().__init__(
            corner,
            position,
            character,
            impassable='#',
            confined_to_board=True
        )
        if orientation is not None:
            self.orientation = orientation
        else:
            self.orientation = np.random.randint(model.N_ORIENTATIONS)
        self.agent_id = agent_id
        self.timeout_duration = timeout_duration
        self.tagged = False


    @property
    def directions(self):
        return [
            self._north,
            self._east,
            self._south,
            self._west
        ]


    @property
    def tagged(self):
        return not self._visible


    @tagged.setter
    def tagged(self, value):
        self._visible = not value
    

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if self.tagged:
            if self.timeout > 0:
                self.timeout -= 1
            else:
                self.tagged = False

        elif actions is not None:
            action = actions.get(self.agent_id)
            y, x = self.position
            
            if action is None:
                self._stay(board, the_plot)
            elif things[model.BEAM_CHAR].curtain[y, x]:
                self.tagged = True
                self.timeout = self.timeout_duration
            elif action in model.STEP_RANGE:
                move = model.move_direction(action, self.orientation)
                self.directions[move](board, the_plot)
            elif action in model.ROTATE_RANGE:
                self.orientation = model.rotate_orientation(self.orientation, action)
            elif action == model.STAND_STILL:
                self._stay(board, the_plot)
        else:
            self._stay(board, the_plot)