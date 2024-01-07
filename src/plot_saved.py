import numpy as np
import evaluation
from main import *



IMG_ROOT = '../img/'
REW_IMGNAME = 'expected_rewards.svg'
EFF_IMGNAME = 'efficiency_score.svg'
SUS_IMGNAME = 'sustainability_score.svg'
EQU_IMGNAME = 'equality_score.svg'
PEA_IMGNAME = 'peace_score.svg'


agent_dirs = [
    CNN_DQN_AGENT_DIR,
    DQN_AGENT_DIR,
    RANDOM_AGENT_DIR
]
map_dirs = [
    OPEN_MAP_DIR,
    # SINGLE_ENTRANCE_MAP_DIR,
    # MULTI_ENTRANCE_MAP_DIR
]


for agent_dir in agent_dirs:
    for map_dir in map_dirs:
        load_dir = ROOT_DIR + agent_dir + map_dir
        total_rewards = np.load(load_dir + REW_FNAME)
        efficiency_score = np.load(load_dir + EFF_FNAME)
        sustainability_score = np.load(load_dir + SUS_FNAME)
        equality_score = np.load(load_dir + EQU_FNAME)
        peace_score = np.load(load_dir + PEA_FNAME)

        dir_path = IMG_ROOT + agent_dir + map_dir
        evaluation.create_directory(dir_path)
        evaluation.plot_rewards(total_rewards, 200, outfile=dir_path + REW_IMGNAME)
        evaluation.plot_efficiency(efficiency_score, 200, outfile=dir_path + EFF_IMGNAME)
        evaluation.plot_sustainability(sustainability_score, 200, outfile=dir_path + SUS_IMGNAME)
        evaluation.plot_equality(equality_score, 200, outfile=dir_path + EQU_IMGNAME)
        evaluation.plot_peacefulness(peace_score, 200, outfile=dir_path + PEA_IMGNAME)