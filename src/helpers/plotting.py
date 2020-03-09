import numpy as np


def moving_average(data, average_every):
    ma = np.cumsum(data, dtype=float)
    ma[average_every:] = ma[average_every:] - ma[:-average_every]
    ma = ma[average_every - 1:] / average_every
    return ma