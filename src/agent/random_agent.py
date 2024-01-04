import numpy as np



class RandomAgent:
    def __init__(self, agent_id, n_actions):
        self.agent_id = agent_id
        self.n_actions = n_actions


    def eval(self):
        return


    def train(self):
        return


    def reset(self, full=False):
        return


    def choose_action(self, observation):
        return np.random.randint(self.n_actions)

    
    def observe_result(self, reward, observation):
        return