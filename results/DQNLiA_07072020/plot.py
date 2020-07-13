import matplotlib.pyplot as plt
import csv
import numpy as np
from cycler import cycler  # for mpl>2.2
from scipy import stats


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

plt.rcParams['axes.prop_cycle'] = cycler(color=CB_color_cycle)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


num_trials = 3
dqnlia0 = [[] for t in range(num_trials)]
dqnlia1 = [[] for t in range(num_trials)]

with open('trial_1_0.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        dqnlia0[0].append(float(row[0]))

with open('trial_2_0.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        dqnlia0[1].append(float(row[0]))

with open('trial_3_0.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        dqnlia0[2].append(float(row[0]))

with open('trial_1_1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        dqnlia1[0].append(float(row[0]))

with open('trial_2_1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        dqnlia1[1].append(float(row[0]))

with open('trial_3_1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        dqnlia1[2].append(float(row[0]))

m = 100
y0_m = np.mean(dqnlia0, axis=0)
y0_s = np.std(dqnlia0, axis=0)
y1_m = np.mean(dqnlia1, axis=0)
y1_s = np.std(dqnlia1, axis=0)
y0 = running_mean(y0_m, m)
y1 = running_mean(y1_m, m)

c0 = 1.98 * np.divide(y0_s, y0_m, out=np.zeros_like(y0_s), where=y0_m!=0)
c1 = 1.98 * np.divide(y1_s, y1_m, out=np.zeros_like(y1_s), where=y1_m!=0)

ci0 = running_mean(c0, m)
ci1 = running_mean(c1, m)
x = range(len(y0))
# #
plt.plot(x, y0)
plt.plot(x, y1)
# #
plt.legend(['DQN [full state]', 'DQNLiA [[self y, ball x, ball y], full state]'], loc='lower right')
plt.fill_between(x, (y0 - ci0), (y0 + ci0), color='b', alpha=.1)
plt.fill_between(x, (y1-ci1), (y1+ci1), color='r', alpha=.1)
# #
plt.show()
