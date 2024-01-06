import numpy as np
import multiprocessing as mp # torch.multiprocessing is only necessary if tensors are passed in queues
from enum import Enum
from copy import copy



class AgentCommand(Enum):
    QUIT = 0
    EVAL = 1
    TRAIN = 2
    RESET = 3
    ACT = 4
    OBSERVE = 5


def agent_process(agent, todo_q, actions_q):
    agent_id = agent.agent_id
    cmd, *args = todo_q.get()

    while cmd != AgentCommand.QUIT:
        if cmd == AgentCommand.EVAL:
            agent.eval()
        elif cmd == AgentCommand.TRAIN:
            agent.train()
        elif cmd == AgentCommand.RESET:
            agent.reset(*args)
        elif cmd == AgentCommand.ACT:
            action = agent.choose_action(*args)
            actions_q.put((agent_id, action))
        elif cmd == AgentCommand.OBSERVE:
            agent.observe_result(*args)
        
        cmd, *args = todo_q.get()


class AgentPool:
    def __init__(self, env, agents, multiprocessing=False):
        self.env = env
        self.multiprocessing = multiprocessing
        self.agent_ids = [agent.agent_id for agent in agents]

        if self.multiprocessing:
            self.actions_queue = mp.Queue(len(agents))
            self.agent_todo_queues = {
                agent.agent_id: mp.Queue()
                for agent in agents
            }

            self.agents = []
            for agent in agents:
                agent_id = agent.agent_id
                todo_q = self.agent_todo_queues[agent_id]
                agent_p = mp.Process(target=agent_process, args=(agent, todo_q, self.actions_queue))
                agent_p.start()
                self.agents.append(agent_p)
        else:
            self.agents = agents

    
    def __del__(self):
        if self.multiprocessing:
            for todo_q in self.agent_todo_queues.values():
                todo_q.put((AgentCommand.QUIT,))
            for agent_p in self.agents:
                agent_p.terminate()


    @property
    def game_over(self):
        return np.all(list(self.terminations.values()))


    def _agent_dict_init(self, value):
        return {
            agent_id: copy(value)
            for agent_id in self.agent_ids
        }


    def eval(self):
        if self.multiprocessing:
            for todo_q in self.agent_todo_queues.values():
                todo_q.put((AgentCommand.EVAL,))
        else:
            for agent in self.agents:
                agent.eval()


    def train(self):
        if self.multiprocessing:
            for todo_q in self.agent_todo_queues.values():
                todo_q.put((AgentCommand.TRAIN,))
        else:
            for agent in self.agents:
                agent.train()


    def reset(self, full=False):
        self.observations, self.infos = self.env.reset()

        if self.multiprocessing:
            for todo_q in self.agent_todo_queues.values():
                todo_q.put((AgentCommand.RESET, full))
        else:
            for agent in self.agents:
                agent.reset(full)

        self.terminations = self._agent_dict_init(False)
        self.truncations = self._agent_dict_init(False)
        self.reward_history = self._agent_dict_init([])
        self.truncation_history = self._agent_dict_init([])


    def step(self):
        if self.multiprocessing:
            acting_agents = 0
            for agent_id, todo_q in self.agent_todo_queues.items():
                if not self.truncations[agent_id]:
                    observation = self.observations[agent_id]
                    todo_q.put((AgentCommand.ACT, observation))
                    acting_agents += 1
            
            actions = dict()
            while len(actions) < acting_agents:
                agent_id, action = self.actions_queue.get()
                actions[agent_id] = action
        else :
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

        if self.multiprocessing:
            for agent_id, todo_q in self.agent_todo_queues.items():
                reward = self.rewards.get(agent_id, 0)
                truncation = self.truncations.get(agent_id, False)
                self.reward_history[agent_id].append(reward)
                self.truncation_history[agent_id].append(truncation)

                observation = self.observations.get(agent_id)
                if not truncation and observation is not None:
                    todo_q.put((AgentCommand.OBSERVE, reward, observation))
        else:
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
        return self.env.render()