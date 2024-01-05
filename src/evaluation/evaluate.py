import numpy as np
from tqdm import tqdm
from .metrics import (
    calculate_utilitarian_metric,
    calculate_sustainability_metric,
    calculate_equality_metric,
    calculate_peace_metric,
)



def evaluate(agent_pool, n_runs, n_episodes):
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
            # Execute episode
            while not agent_pool.game_over:
                agent_pool.step()

            # Calculate metric scores
            rewards = np.array(list(agent_pool.reward_history.values()))
            truncations = np.array(list(agent_pool.truncation_history.values()))

            efficiency_score[run, ep] = calculate_utilitarian_metric(rewards)
            sustainability_score[run, ep] = calculate_sustainability_metric(rewards)
            equality_score[run, ep] = calculate_equality_metric(rewards)
            peace_score[run, ep] = calculate_peace_metric(truncations)

            agent_pool.reset()

    return (
        efficiency_score,
        sustainability_score,
        equality_score,
        peace_score
    )