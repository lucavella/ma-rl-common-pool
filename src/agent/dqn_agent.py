import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import agent.dqn as dqn



EPSILON_START = 1
EPSILON_END = 0.1
ALPHA = 1e-4
GAMMA = 0.99
BATCH_SIZE = 32
UPDATE_INTERVAL = 4
TARGET_UPDATE_INTERVAL = 1000



class DQNAgent:
    def __init__(
        self,
        agent_id,
        n_actions,
        n_episodes,
        device,
        observation_shape,
        cnn_dqn=True,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        learning_rate=ALPHA,
        discount_factor=GAMMA,
        batch_size=BATCH_SIZE,
        update_interval=UPDATE_INTERVAL,
        target_update_interval=TARGET_UPDATE_INTERVAL,
    ):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.n_episodes = n_episodes
        self.device = device
        self.observation_shape = observation_shape
        self.cnn_dqn = True
        self.epsilon_start = epsilon_start
        self.epsilon_step = (epsilon_end - epsilon_start) / n_episodes
        self.learning_rate = learning_rate
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.learn = True
        self.criterion = nn.SmoothL1Loss()


    def eval(self):
        self.learn = False
        if hasattr(self, 'policy_net') and hasattr(self, 'target_net'):
            self.policy_net.eval()
            self.target_net.eval()


    def train(self):
        self.learn = True
        if hasattr(self, 'policy_net') and hasattr(self, 'target_net'):
            self.policy_net.eval()
            self.target_net.eval()


    def reset(self, full=False):
        if full:
            self.episode = 0
            self.epsilon = self.epsilon_start
            self.memory = dqn.ReplayMemory(self.n_episodes)
            if self.cnn_dqn:
                self.policy_net = dqn.CnnDQN(self.observation_shape, self.n_actions).to(self.device)
                self.target_net = dqn.CnnDQN(self.observation_shape, self.n_actions).to(self.device)
            else:
                self.policy_net = dqn.DQN(self.observation_shape, self.n_actions).to(self.device)
                self.target_net = dqn.DQN(self.observation_shape, self.n_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.learn:
                self.policy_net.eval()
                self.target_net.eval()
            else:
                self.policy_net.train()
                self.target_net.train
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.alpha, amsgrad=True)
            # self.optimizer = optim.RMSProp(self.policy_net.parameters(), lr=self.alpha)
        elif self.learn:
            self.episode += 1
            self.epsilon -= self.epsilon_step


    def choose_action(self, observation):
        self.last_observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0) / 256
        if self.learn and random.random() < self.epsilon:
            self.last_action = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                # kan zijn dat het target_net is (bron human-level ergens bij kleine letters)
                self.last_action = self.policy_net(self.last_observation).max(1).indices.view(1, 1) 

        return self.last_action


    # 1) EXPERIENCE REPLAY:
    #     randomizes over the data, thereby removing correlations in the observation sequence and smoothing over changes in the data distribution
    # 2) Iterative update:
    #     adjust the action-values (Q) towards target values that are only periodically updated, thereby reducing the correlations with the target
    def observe_result(self, reward, observation):
        if self.learn:
            next_observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0) / 256

            # Store the transition in memory
            # memory = (state, action, next_state, reward)
            self.memory.push(self.last_observation, self.last_action, next_observation, reward)

            # if self.episode % self.update_interval == 0:
            self.optimize_model()

            """
            More precisely, every C updates we clone the network Q to obtain a target network ^Q and use ^Q for generating the
            Q-learning targets yj for the following C updates to Q
            Source: human-level control
            """
            if self.episode % self.target_update_interval == 0:
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]
                    self.target_net.load_state_dict(target_net_state_dict)


    # Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            # load memory first
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = dqn.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action).to(self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward).to(self.device).unsqueeze(1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()