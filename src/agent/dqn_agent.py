import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dqn import DQN, ReplayMemory, optimize_model



GAMMA = 0.99
LR = 1e-4
# LR = 0.00025 # from paper human action
BATCH_SIZE = 32

EPSILON_START = 1
EPSILON_END = 0.1



class DQNAgent:
    def __init__(
        self,
        agent_id,
        n_actions,
        device,
        observation_shape,
        epsilon_start,
        epsilon_step,
        n_episodes,
        learning_rate,
        discount_factor,
        batch_size,
        update_frequency,
        target_update_frequency,
    ):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.device = device
        self.observation_shape = observation_shape
        self.epsilon_start = epsilon_start
        self.epsilon_step = epsilon_step
        self.learning_rate = learning_rate
        self.n_episodes = n_episodes
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.update_frequency = update_frequency # 4, source human-level
        self.target_update_frequency = target_update_frequency # 10, 1000 in source https://github.com/jihoonerd/Human-level-control-through-deep-reinforcement-learning/blob/master/dqn/agent/dqn_agent.py)
        self.learn = True


    def eval(self):
        self.learn = False


    def train(self):
        self.learn = True


    def reset(self, full=False):
        if full:
            self.episode = 0
            self.epsilon = self.epsilon_start
            self.memory = dqn.ReplayMemory(self.n_episodes)
            input_size = np.prod(self.observation_shape)
            self.policy_net = dqn.DQN(input_size).to(self.device)
            self.target_net = dqn.DQN(input_size).to(self.device)
            self.target_net.load_state_dict(policy_net.state_dict())
            self.optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
            # self.optimizer = optim.RMSProp(policy_net.parameters(), lr=LR,
        elif self.learn:
            self.episode += 1
            self.epsilon -= self.epsilon_step


    def choose_action(self, observation):
        self.last_observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) / 256
        if self.learn and random.random() < self.epsilon:
            self.last_action = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                # kan zijn dat het target_net is (bron human-level ergens bij kleine letters)
                self.last_action = self.policy_net(self.last_observation).max(1).indices.view(1, 1) 


    # 1) EXPERIENCE REPLAY:
    #     randomizes over the data, thereby removing correlations in the observation sequence and smoothing over changes in the data distribution
    # 2) Iterative update:
    #     adjust tge action-values (Q) towards target values that are only periodically updated, thereby reducing the correlations with the target
    #     Dit betekend dat de target DQN pas om de zoveel iteraties wordt geupdate. Het training DQN elke keer (zo neem ik het aan)
    def observe_result(self, reward, observation):
        if self.learn:
            next_observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) / 256

            # Store the transition in memory
            # memory = ('state', 'action', 'next_state', 'reward'))
            self.memory.push(self.last_observation, action, next_observation, reward)

            # ik weet niet of dit elke episode moet of ook om de zoveel (in human level is dit denk ik 4)
            #if (i_episode % update_frequency == 0):
            dqn.optimize_model(self.device, self.policy_net, self.target_net, self.memory, self.gamma, self.batch_size)

            """
            More precisely, every C updates we clone the network Q to obtain a target network ^Q and use ^Q for generating the
            Q-learning targets yj for the following C updates to Q
            source (human-level control)
            """
            if (self.episode % self.target_update_freq == 0):
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]
                    self.target_net.load_state_dict(target_net_state_dict)