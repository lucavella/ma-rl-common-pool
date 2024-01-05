import numpy as np



def calculate_utilitarian_metric(rewards):
    duration = rewards.shape[1]
    acc_rewards = np.sum(rewards, axis=1)
    return np.mean(acc_rewards / duration)


def calculate_sustainability_metric(rewards):
    positive_reward_times = np.apply_along_axis(
        lambda x: np.mean(np.where(x > 0)),
        axis=1,
        arr=rewards
    )
    return np.mean(positive_reward_times[~np.isnan(positive_reward_times)])


def calculate_equality_metric(rewards):
    n_agents = rewards.shape[0]
    acc_rewards = np.sum(rewards, axis=1)
    acc_rewards_diff = np.abs(acc_rewards[:, np.newaxis] - acc_rewards)
    tot_acc_rewards_diff = np.sum(acc_rewards_diff)
    tot_rewards = np.sum(acc_rewards)
    return tot_acc_rewards_diff / (2 * n_agents * tot_rewards)


def calculate_peace_metric(truncations):
    n_agents, duration = truncations.shape
    timeouts = np.sum(truncations)
    return (n_agents * duration - timeouts) / duration