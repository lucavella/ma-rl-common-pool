import numpy as np
from tqdm import tqdm
from PIL import Image
from .metrics import (
    calculate_utilitarian_metric,
    calculate_sustainability_metric,
    calculate_equality_metric,
    calculate_peace_metric,
)
from .utils import create_directory



def evaluate(agent_pool, n_runs, n_episodes, save_dir=None, episode_save_interval=1000):
    def save_episode(run, episode):
        save_path = f'{save_dir}/{run}/{episode}/'
        create_directory(save_path)
        frame_i = 0
        agent_pool.eval()

        while not agent_pool.game_over:
            frame = agent_pool.render()
            img = Image.fromarray(frame.astype(np.uint8), mode='RGB')
            img.save(save_path + f'frame_{frame_i}.png')
            agent_pool.step()
            frame_i += 1

        frame = agent_pool.render()
        img = Image.fromarray(frame.astype(np.uint8), mode='RGB')
        img.save(save_path + f'frame_{frame_i}.png')
        agent_pool.reset()
        agent_pool.train()
    
    n_agents = len(agent_pool.agent_ids)
    expected_rewards = np.zeros((n_runs, n_episodes))
    efficiency_score = np.zeros((n_runs, n_episodes))
    sustainability_score = np.zeros((n_runs, n_episodes))
    equality_score = np.zeros((n_runs, n_episodes))
    peace_score = np.zeros((n_runs, n_episodes))

    agent_pool.train()

    # Do n_runs independent runs
    for run in tqdm(range(n_runs), desc="RUN", leave=False):
        agent_pool.reset(full=True)

        # Do n_episodes episodes
        for ep in tqdm(range(n_episodes), desc="EPISODE", leave=False):
            if save_dir and ep % episode_save_interval == 0:
                save_episode(run, ep)

            # Execute episode
            while not agent_pool.game_over:
                agent_pool.step()

            # Calculate metric scores
            rewards = np.array(list(agent_pool.reward_history.values()))
            truncations = np.array(list(agent_pool.truncation_history.values()))

            expected_rewards[run, ep] = np.mean(rewards)
            efficiency_score[run, ep] = calculate_utilitarian_metric(rewards)
            sustainability_score[run, ep] = calculate_sustainability_metric(rewards)
            equality_score[run, ep] = calculate_equality_metric(rewards)
            peace_score[run, ep] = calculate_peace_metric(truncations)

            agent_pool.reset()

        if save_dir:
            save_episode(run, n_episodes)
            
    return (
        expected_rewards,
        efficiency_score,
        sustainability_score,
        equality_score,
        peace_score
    )