import matplotlib.pyplot as plt
import csv
import numpy


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

m = 30
plt.plot(running_mean(dqn, m))
plt.plot(running_mean(dqnlia0, m))
plt.plot(running_mean(dqnlia1, m))
plt.legend(['DQN', 'DQNLiA - 1 sub', 'DQNLiA - 2 subs'])
plt.show()
