import numpy as np
from numpy import genfromtxt
import sys

optimal_Q = genfromtxt('q_table_taxi.csv', delimiter=',')

print(optimal_Q)


def optimal_policy(state):
    return np.argmax(optimal_Q[state])