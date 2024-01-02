import numpy as np



def calculate_utilitarian_metric(duration, *rewards):
    acc_rewards = np.sum(rewards, axis=1)
    return np.mean(acc_rewards / duration)


def calculate_equality_metric(*rewards):
    acc_rewards = np.sum(rewards, axis=1)
    acc_rewards_diff = np.abs(acc_rewards[:, np.newaxis] - acc_rewards)
    tot_acc_rewards_diff = np.sum(acc_rewards_diff)
    tot_rewards = np.sum(acc_rewards)
    return tot_acc_rewards_diff / (2 * len(rewards) * tot_rewards)


def calculate_sustainability_metric(*rewards):
    positive_reward_times = np.apply_along_axis(
        lambda x: np.where(x > 0),
        axis=0,
        arr=rewards
    )[:, 0, :]
    return np.mean(positive_reward_times)


def calculate_peace_metric(duration, *observations):
    timeouts = np.sum(is_timeout(observations))
    return (len(observations) * duration - timeouts) / duration