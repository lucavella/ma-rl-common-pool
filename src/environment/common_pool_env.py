import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
from pycolab.ascii_art import ascii_art_to_game, Partial
from pycolab.rendering import ObservationToArray
import matplotlib.pyplot as plt
from string import ascii_letters
from functools import lru_cache, cached_property
from copy import copy
import model
import model.things as things



OBSERVE_WIDTH_VIEW = 21
OBSERVE_FRONT_VIEW = 20

RESPAWN_BALL_RADIUS = 2
PROB_L_1_2 = .01
PROB_L_3_4 = .05
PROB_L_GT_4 = .1
RESPAWN_PROBABILITIES = [PROB_L_1_2, PROB_L_3_4, PROB_L_GT_4]

BEAM_RANGE = 10
BEAM_WIDTH = 5
TIME_OUT = 25

EMPTY_COLOR = (0, 0, 0)
WALL_COLOR  = (127, 127, 127)
APPLE_COLOR = (0, 255, 0)
BEAM_COLOR  = (255, 255, 0)
AGENT_COLOR = (255, 0, 0)
POV_COLOR   = (0, 0, 255)


class CommonPoolEnv(ParallelEnv):
    metadata = {
        'name': 'common_pool_environment',
        'render_modes': ['human', 'rgb_array']
    }

    def __init__(
        self,
        env_map,
        agent_ids,
        max_steps,
        observation_ahead=OBSERVE_FRONT_VIEW,
        observation_width=OBSERVE_WIDTH_VIEW,
        agent_timeout=TIME_OUT,
        apple_respawn_radius=RESPAWN_BALL_RADIUS,
        apple_respawn_probabilities=RESPAWN_PROBABILITIES,
        beam_width=BEAM_WIDTH,
        beam_range=BEAM_RANGE,
        render_mode=None,
        render_speed=.05,
    ):
        self.env_map = env_map
        self.possible_agents = agent_ids
        self.n_agents = len(agent_ids)
        self.agents_char = list(ascii_letters)[:self.n_agents]
        self.observation_ahead = observation_ahead
        self.observation_width = observation_width
        self.agent_timeout = agent_timeout
        self.max_steps = max_steps
        self.apple_respawn_radius = apple_respawn_radius
        self.apple_respawn_probabilities = apple_respawn_probabilities
        self.beam_width = beam_width
        self.beam_range = beam_range
        self.render_speed = render_speed
        self.render_mode = render_mode

        self.engine = None
        self.ota = ObservationToArray(value_mapping=self.color_map)


    def _agent_dict_init(self, value):
        return {
            agent_id: copy(value)
            for agent_id in self.agents
        }

    
    @cached_property
    def color_map(self):
        color_map = {
            model.EMPTY_CHAR: EMPTY_COLOR,
            model.WALL_CHAR: WALL_COLOR,
            model.APPLE_CHAR: APPLE_COLOR,
            model.BEAM_CHAR: BEAM_COLOR,
            **{
                agent_char: AGENT_COLOR
                for agent_char in self.agents_char
            }
        }

        return color_map


    def _agent_observation(self, agent_id):
        agent_char = self.agents_char[agent_id]
        observation_array = self.ota(self._observation)
        agent = self.engine.things[agent_char]
        counter_clock_k = - agent.orientation % model.N_ORIENTATIONS
        rotated_observation = np.rot90(observation_array, k=agent.orientation, axes=(1, 2))

        _, h, w = rotated_observation.shape
        y, x = model.rotate_position(agent.position, agent.orientation, observation_array.shape[1:])

        x_min = x - self.observation_width // 2
        x_max = x_min + self.observation_width
        y_max = y + 1
        y_min = y_max - self.observation_ahead

        x_left_bound = max(x_min, 0)
        x_right_bound = min(x_max, w)
        y_top_bound = max(y_min, 0)
        cropped_observation = rotated_observation[:, y_top_bound:y_max, x_left_bound:x_right_bound]

        # More than double as fast as np.pad
        left_pad = x_left_bound - x_min
        right_pad = x_max - x_right_bound
        top_pad = y_top_bound - y_min
        left_padding = np.full((self.observation_ahead - top_pad, left_pad, 3), EMPTY_COLOR).transpose(2, 0, 1)
        right_padding = np.full((self.observation_ahead - top_pad, right_pad, 3), EMPTY_COLOR).transpose(2, 0, 1)
        top_padding = np.full((top_pad, self.observation_width, 3), EMPTY_COLOR).transpose(2, 0, 1)
        width_padded_observation = np.append(np.append(left_padding, cropped_observation, axis=2), right_padding, axis=2)
        padded_observation = np.append(top_padding, width_padded_observation, axis=1)

        # Due to z-order, current agent might be under another
        # So this cannot be done with a colormap
        if not agent.tagged:
            x = self.observation_width // 2
            y = self.observation_ahead - 1
            padded_observation[:, y, x] = POV_COLOR

        return padded_observation


    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(0, 255, shape=(3, self.observation_ahead, self.observation_width), dtype=int)


    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(model.N_ACTIONS)


    @property
    def observations(self):
        return {
            agent_id: self._agent_observation(agent_id)
            for agent_id, agent_char in zip(self.agents, self.agents_char)
        }


    def reset(self):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        new_layout = self.env_map.initialize(self.agents_char)

        # Documentation: https://github.com/google-deepmind/pycolab/blob/6504c8065dd5322438f23a2a9026649904f3322a/pycolab/ascii_art.py#L31
        self.engine = ascii_art_to_game(
            new_layout,
            what_lies_beneath=model.EMPTY_CHAR,
            sprites={
                agent_char: Partial(things.AgentSprite, agent_id, self.agent_timeout)
                for agent_id, agent_char in zip(self.agents, self.agents_char)
            }, drapes={
                model.APPLE_CHAR: Partial(
                    things.AppleDrape,
                    self.agents,
                    self.agents_char,
                    self.apple_respawn_radius,
                    self.apple_respawn_probabilities),
                model.BEAM_CHAR: Partial(
                    things.BeamDrape,
                    self.agents,
                    self.agents_char,
                    self.beam_width,
                    self.beam_range)
            }, update_schedule=[
                model.BEAM_CHAR, *self.agents_char, model.APPLE_CHAR
            ], z_order=[
                model.APPLE_CHAR, *self.agents_char, model.BEAM_CHAR
            ]
        )
        self._observation, _, _ = self.engine.its_showtime()

        self.infos = self._agent_dict_init(dict())
        self.terminations = self._agent_dict_init(False)

        return self.observations, self.infos


    def render(self):
        observation_array = self.ota(self._observation)
        if self.render_mode == 'human':
            plt.imshow(observation_array.transpose(1, 2, 0))
            plt.axis('off')
            plt.show(block=False)
            plt.pause(self.render_speed)
        elif self.render_mode == 'rgb_array':
            return observation_array.transpose(1, 2, 0)


    def step(self, actions):
        if len(self.agents) == 0:
            return

        self._observation, self.rewards, _ = self.engine.play(actions)

        self.truncations = {
            agent_id: self.engine.things[agent_char].tagged
            for agent_id, agent_char in zip(self.agents, self.agents_char)
        }
    
        # apples_left = self.engine.things[model.APPLE_CHAR].any_left
        if self.timestep >= self.max_steps: # or not apples_left:
            self.truncations = self._agent_dict_init(True)
            self.terminations = self._agent_dict_init(True)
            self.agents.clear()
        self.timestep += 1

        infos = self._agent_dict_init(dict())
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos