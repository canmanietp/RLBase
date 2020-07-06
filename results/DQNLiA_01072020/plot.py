import matplotlib.pyplot as plt
import csv
import numpy
from cycler import cycler  # for mpl>2.2

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

plt.rcParams['axes.prop_cycle'] = cycler(color=CB_color_cycle)

def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


dqn = []
dqnlia0 = []
dqnlia1 = []

with open('trial_1_dqn0.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        dqn.append(float(row[0]))

with open('trial_1_lia0.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        dqnlia0.append(float(row[0]))

with open('trial_1_lia1.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        dqnlia1.append(float(row[0]))

m = 100
plt.plot(running_mean(dqn, m))
plt.plot(running_mean(dqnlia0, m))
plt.plot(running_mean(dqnlia1, m))
plt.legend(['DQN', 'DQNLiA [[self y, ball x, ball y]]', 'DQNLiA [[ball y], [self y, ball x, ball y]]'])
plt.show()
