import numpy as np
from copy import copy



class AgentPool:
    def __init__(self, env, agents):
        self.agents = agents
        self.env = env


    @property
    def game_over(self):
        return np.all(list(self.terminations.values()))


    @property
    def agent_id(self):
        return [
            agent.agent_id
            for agent in self.agents
        ]


    def _agent_dict_init(self, value):
        return {
            agent.agent_id: copy(value)
            for agent in self.agents
        }


    def eval(self):
        for agent in self.agents:
            agent.eval()


    def train(self):
        for agent in self.agents:
            agent.train()


    def reset(self, full=False):
        self.observations, self.infos = self.env.reset()
        for agent in self.agents:
            agent.reset(full)

        self.terminations = self._agent_dict_init(False)
        self.truncations = self._agent_dict_init(False)
        self.reward_history = self._agent_dict_init([])
        self.truncation_history = self._agent_dict_init([])


    def step(self):
        actions = {
            agent.agent_id: agent.choose_action(self.observations[agent.agent_id])
            for agent in self.agents
            if not self.truncations[agent.agent_id]
        }
        
        ( self.observations,
          self.rewards,
          self.terminations,
          self.truncations,
          self.infos ) = self.env.step(actions)

        for agent in self.agents:
            agent_id = agent.agent_id
            reward = self.rewards.get(agent_id, 0)
            truncation = self.truncations.get(agent_id, False)
            self.reward_history[agent_id].append(reward)
            self.truncation_history[agent_id].append(truncation)

            observation = self.observations.get(agent_id)
            if not truncation and observation is not None:
                agent.observe_result(reward, observation)

    
    def render(self):
        self.env.render()