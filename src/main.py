import warnings
import os
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import environment
import model
import agent
import evaluation

warnings.filterwarnings('ignore')



N_RUNS = 1
N_EPISODES = 5_000
MAX_STEPS = 500

ROOT_DIR = '../results/'
RANDOM_AGENT_DIR = 'random/'
CNN_DQN_AGENT_DIR = 'cnn_dqn/'
DQN_AGENT_DIR = 'dqn/'
OPEN_MAP_DIR = 'open_map/'
SINGLE_ENTRANCE_MAP_DIR = 'single_entrance_map/'
MULTI_ENTRANCE_MAP_DIR = 'multi_entrance_map/'
REW_FNAME = 'expected_rewards.npy'
EFF_FNAME = 'efficiency_score.npy'
SUS_FNAME = 'sustainability_score.npy'
EQU_FNAME = 'equality_score.npy'
PEA_FNAME = 'peace_score.npy'


if __name__ == '__main__':
    mp.set_start_method('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent_ids = list(range(10))
    observation_shape = (3, environment.OBSERVE_FRONT_VIEW, environment.OBSERVE_WIDTH_VIEW)

    learning_agents = [
        (agent.DQNAgent, CNN_DQN_AGENT_DIR, N_EPISODES, device, observation_shape, True),
        (agent.DQNAgent, DQN_AGENT_DIR, N_EPISODES, device, observation_shape, False),
        (agent.RandomAgent, RANDOM_AGENT_DIR),
    ]
    map_layouts = [
        (model.OPEN_MAP, OPEN_MAP_DIR),
        (model.SINGLE_ENTRANCE_MAP, SINGLE_ENTRANCE_MAP_DIR),
        (model.MULTI_ENTRANCE_MAP, MULTI_ENTRANCE_MAP_DIR)
    ]

    for agent_class, agent_dir, *args in tqdm(learning_agents, desc='ALGOS', leave=False):
        for map_layout, map_dir in tqdm(map_layouts, desc='ENVS', leave=False):
            dir_path = ROOT_DIR + agent_dir + map_dir
            evaluation.create_directory(dir_path)

            game_env = environment.CommonPoolEnv(map_layout, agent_ids, MAX_STEPS, render_mode='rgb_array')

            pool = agent.AgentPool(game_env, [
                agent_class(agent_id, model.N_ACTIONS, *args)
                for agent_id in agent_ids
            ], multiprocessing=True)

            ( total_rewards,
            efficiency_score,
            sustainability_score,
            equality_score,
            peace_score ) = evaluation.evaluate(pool, N_RUNS, N_EPISODES, dir_path)

            np.save(dir_path + REW_FNAME, arr=total_rewards)
            np.save(dir_path + EFF_FNAME, arr=efficiency_score)
            np.save(dir_path + SUS_FNAME, arr=sustainability_score)
            np.save(dir_path + EQU_FNAME, arr=equality_score)
            np.save(dir_path + PEA_FNAME, arr=peace_score)