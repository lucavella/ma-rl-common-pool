import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_rewards(rewards_results, rolling_mean=None, outfile=None):
    plot_episode_results(
        rewards_results,
        ylabel="Expected reward",
        rolling_mean=rolling_mean,
        outfile=outfile
    )


def plot_efficiency(efficiency_results, rolling_mean=None, outfile=None):
    plot_episode_results(
        efficiency_results,
        ylabel="Efficiency (U)",
        rolling_mean=rolling_mean,
        outfile=outfile
    )


def plot_sustainability(sustainability_results, rolling_mean=None, outfile=None):
    plot_episode_results(
        sustainability_results,
        ylabel="Sustainability (S)",
        rolling_mean=rolling_mean,
        outfile=outfile
    )


def plot_equality(equality_results, rolling_mean=None, outfile=None):
    plot_episode_results(
        equality_results,
        ylabel="Equality (E)",
        rolling_mean=rolling_mean,
        outfile=outfile
    )


def plot_peacefulness(peacefulness_results, rolling_mean=None, outfile=None):
    plot_episode_results(
        peacefulness_results,
        ylabel="Peacefulness (P)",
        rolling_mean=rolling_mean,
        outfile=outfile
    )


# Common code used in all plots
def plot_episode_results(training_results, ylabel, rolling_mean=None, outfile=None):
    # Training results
    mean_tr = pd.Series(np.mean(training_results, axis=0))
    std_tr = pd.Series(np.std(training_results, axis=0))

    if rolling_mean:
        mean_tr = mean_tr.rolling(rolling_mean)\
                         .mean()\
                         .shift(-rolling_mean)\
                         .dropna()
        std_tr = std_tr.rolling(rolling_mean)\
                       .mean()\
                       .shift(-rolling_mean)\
                       .dropna()
        print(std_tr)

    # Plot the results
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_ylabel(ylabel, size=12)
    ax.set_xlabel("Episode", size=12)
    ax.plot(mean_tr, color='b')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=mean_tr.max() * 1.2)
    # ax.fill_between(mean_tr - std_tr, mean_tr + std_tr, color='b', alpha=0.2)
    fig.tight_layout()

    # Save plot if necessary
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()