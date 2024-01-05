import numpy as np
import matplotlib.pyplot as plt



def plot_efficiency(efficiency_results, outfile=None):
    plot_episode_results(
        efficiency_results,
        ylabel="Efficiency (U)",
        outfile=outfile
    )


def plot_sustainability(sustainability_results, outfile=None):
    plot_episode_results(
        sustainability_results,
        ylabel="Sustainability (S)",
        outfile=outfile
    )


def plot_equality(equality_results, outfile=None):
    plot_episode_results(
        equality_results,
        ylabel="Equality (E)",
        outfile=outfile
    )


def plot_peacefulness(peacefulness_results, outfile=None):
    plot_episode_results(
        peacefulness_results,
        ylabel="Peacefulness (P)",
        outfile=outfile
    )


# Common code used in all plots
def plot_episode_results(training_results, ylabel, outfile=None):
    # Training results
    mean_tr = np.mean(training_results, axis=0)
    std_tr = np.std(training_results, axis=0)

    # X-axis values
    x_tr = np.arange(0, len(mean_tr))

    # Plot the results
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylabel(ylabel, size=12)
    ax.set_xlabel("Episode", size=12)
    ax.plot(x_tr, mean_tr, color='b')
    ax.fill_between(x_tr, mean_tr - std_tr, mean_tr + std_tr, color='b', alpha=0.2)
    fig.tight_layout()

    # Save plot if necessary
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()