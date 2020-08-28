import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt
import csv
import numpy as np
from cycler import cycler  # for mpl>2.2
import pandas as pd

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

plt.rcParams['axes.prop_cycle'] = cycler(color=CB_color_cycle)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_trials(filename, num_trials, algs, moving_average=None, x_lim=None, y_lim=None):
    rewards = []

    for t in range(num_trials):
        with open(filename.format(t + 1), 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            row_tracker = []
            for ep, row in enumerate(plots):
                if moving_average:
                    row_tracker.append([float(r) for r in row[0:len(algs)]])
                    if ep % moving_average == 0:
                        for alg in range(len(row) - 1):
                            rewards.append([t, ep, algs[alg],
                                            float(np.average([el[alg] for el in row_tracker][-moving_average:]))])
                else:
                    for alg in range(len(row) - 1):
                        rewards.append([t, ep, algs[alg], float(row[alg])])

    restructure = pd.DataFrame(rewards)
    restructure.columns = ["trial", "episode", "alg", "reward"]
    sns.set_style("white")
    sns.lineplot(x="episode", y="reward", hue="alg", data=restructure, err_style="bars", ci=95)
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)

    plt.legend(algs)
    plt.show()

#
# directory = '2020-08-05 12:57:48_taxi/trial_{}.csv'
# num_trials = 10
# algs = ['MaxQ', 'Q', 'LOARA_known', 'LOARA_unknown']
# moving_average = 50
# x_lim = [0, 900]
# y_lim = [-200, 10]
directory = '2020-08-05 14:17:56_taxilarge/trial_{}.csv'
num_trials = 10
algs = ['MaxQ', 'Q', 'LOARA_known', 'LOARA_unknown']
moving_average = 100
x_lim = [0, 4900]
y_lim = [-1000, 10]
plot_trials(directory, num_trials, algs, moving_average, x_lim, y_lim)
