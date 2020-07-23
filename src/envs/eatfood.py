import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import gym
import numpy as np

nR = 5  # number of rows
nC = 5  # number of columns


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class EatFoodEnv(discrete.DiscreteEnv):
    """
    Actions:
    There are 4 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    """

    def __init__(self):
        nS = (nR*nC)**3  # rabbit locations x food locations x wolf locations
        maxR = nR - 1
        maxC = nC - 1
        nA = 4

        isd = np.zeros(nS)
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        for row in range(nR):
            for col in range(nC):
                    for food_idx in range(nR*nC):
                        for wolf_idx in range(nR*nC):
                            state = self.encode(row, col, food_idx, wolf_idx)

                            for act in range(nA):
                                newrow, newcol, newfood_idx, newwolf_idx = row, col, food_idx, wolf_idx
                                done = False
                                reward = 0.5  # reward every time step

                                # if wolf caught rabbit, episode ends
                                if (row * nR + col) == wolf_idx:
                                    reward = -20
                                    done = True
                                else:
                                    isd[state] += 1
                                    # for all the actions
                                    if act == 0:  # south
                                        newrow = min(row+1, maxR)
                                    elif act == 1:  # north
                                        newrow = max(row-1, 0)
                                    elif act == 2:  # east
                                        newcol = min(col+1, maxC)
                                    elif act == 3:  # west
                                        newcol = max(col-1, 0)

                                    if (newrow * nR + newcol) == food_idx:
                                        reward = 10  # reward for eating food

                                    wolfcol = wolf_idx % nC
                                    wolfrow = int((wolf_idx - wolfcol) / nR)
                                    newwolfcol = wolfcol
                                    newwolfrow = wolfrow

                                    # wolf moves toward rabbit every time step
                                    if wolfrow == newrow:  # if wolf and rabbit are in same row
                                        if wolfcol > newcol:
                                            newwolfcol = max(wolfcol - 1, 0)
                                        else:
                                            newwolfcol = min(wolfcol + 1, maxC)
                                    else:  # move up or down to get closer to rabbit
                                        if wolfrow > newrow:
                                            newwolfrow = max(wolfrow - 1, 0)
                                        else:
                                            newwolfrow = min(wolfrow + 1, maxR)

                                    newwolf_idx = newwolfrow * nR + newwolfcol

                                newstate = self.encode(newrow, newcol, newfood_idx, newwolf_idx)
                                P[state][act].append((1.0, newstate, reward, done))

        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]

        # Food appears in random location if eaten
        if r == 10:  # dangerously hardcoded here, change value if reward changes
            rabrow, rabcol, food_idx, wolf_idx = self.decode(s)
            s = self.encode(rabrow, rabcol, np.random.randint(nR * nC), wolf_idx)

        # Every other step, wolf does not move
        # Comment this out for a faster wolf
        # if self.step_count % 2 == 0:
        #     rabrow, rabcol, food_idx, ex_wolf_idx = self.decode(self.s)
        #     rabrow, rabcol, food_idx, wolf_idx = self.decode(s)
        #     s = self.encode(rabrow, rabcol, food_idx, ex_wolf_idx)

        self.s = s
        self.lastaction = a
        return s, r, d, {"prob": p}

    def encode(self, rabrow, rabcol, food_idx, wolf_idx):
        # (5) 5, 25, 25
        # (nR) nC, 25, 25
        i = rabrow
        i *= nC  # number of rab col ids
        i += rabcol
        i *= nC*nR  # number of food ids
        i += food_idx
        i *= nC*nR  # number of wolf ids
        i += wolf_idx
        #if i < 0 : print(rabrow, rabcol, food_idx, wolf_idx)
        return i

    def decode(self, i):
        out = []
        out.append(i % (nC*nR))
        i = i // (nC*nR)
        out.append(i % (nC*nR))
        i = i // (nC*nR)
        out.append(i % nR)
        i = i // nR
        out.append(i)
        assert 0 <= i < nR
        return reversed(out)



