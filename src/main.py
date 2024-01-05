import warnings
import os
import environment
import model
import agent
import evaluation

warnings.filterwarnings('ignore')



N_RUNS = 5
N_EPISODES = 10_000
MAX_STEPS = 1000

ROOT_DIR = '../img/'
RANDOM_AGENT_DIR = 'random/'
DQN_AGENT_DIR = 'dqn/'
OPEN_MAP_DIR = 'open_map/'
EFF_FNAME = 'efficiency.svg' 
SUS_FNAME = 'sustainability.svg'
EQU_FNAME = 'equality.svg'
PEA_FNAME = 'peacefulness.svg'


if __name__ == '__main__':
    agent_ids = list(range(10))
    game_env = environment.CommonPoolEnv(model.OPEN_MAP, agent_ids, MAX_STEPS)
    pool = agent.AgentPool(game_env, [
        agent.RandomAgent(agent_id, model.N_ACTIONS)
        for agent_id in agent_ids
    ])

    ( efficiency_score,
      sustainability_score,
      equality_score,
      peace_score ) = evaluation.evaluate(pool, N_RUNS, N_EPISODES)


    dir_path = ROOT_DIR + RANDOM_AGENT_DIR + OPEN_MAP_DIR
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    evaluation.plot_efficiency(efficiency_score, outfile=dir_path + EFF_FNAME)
    evaluation.plot_sustainability(sustainability_score, outfile=dir_path + SUS_FNAME)
    evaluation.plot_equality(equality_score, outfile=dir_path + EQU_FNAME)
    evaluation.plot_peacefulness(peace_score, outfile=dir_path + PEA_FNAME)