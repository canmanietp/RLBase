import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np
import random

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


class TaxiFuelEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drive to the passenger's location, pick up the passenger, drive to the passenger's destination (another one of the four specified locations), and then drop off the passenger. Once the passenger is dropped off, the episode ends.

    Observations:
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is the taxi), and 4 destination locations.

    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: dropoff passenger

    Rewards:
    There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.


    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')

        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.fuel_loc = (3, 2)

        nS = 7000
        nR = self.num_rows = 5
        nC = self.num_columns = 5
        nF = self.fuel_levels = 14
        maxR = nR - 1
        maxC = nC - 1
        isd = np.zeros(nS)
        nA = 7
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for row in range(nR):
            for col in range(nC):
                for pass_idx in range(len(locs) + 1):
                    for dest_idx in range(len(locs)):
                        for fuel in range(nF):
                            state = self.encode(row, col, pass_idx, dest_idx, fuel)
                            if pass_idx < 4 and pass_idx != dest_idx:
                                isd[state] += 1
                            for a in range(nA):
                                new_row, new_col, new_pass_idx, new_fuel = row, col, pass_idx, fuel
                                reward = -1
                                done = False
                                taxiloc = (row, col)

                                new_fuel = fuel - 1

                                if a == 0:
                                    new_row = min(row + 1, maxR)
                                elif a == 1:
                                    new_row = max(row - 1, 0)
                                elif a == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                    new_col = min(col + 1, maxC)
                                elif a == 3 and self.desc[1 + row, 2 * col] == b":":
                                    new_col = max(col - 1, 0)
                                elif a == 4:  # pickup
                                    if pass_idx != 4 and taxiloc == locs[pass_idx]:
                                        new_pass_idx = 4
                                    else:
                                        reward = -10
                                elif a == 5:  # dropoff
                                    if (taxiloc == locs[dest_idx]) and pass_idx == 4:
                                        new_pass_idx = dest_idx
                                        done = True
                                        reward = 20
                                    elif (taxiloc in locs) and pass_idx == 4:
                                        new_pass_idx = locs.index(taxiloc)
                                        reward = -10
                                    else:
                                        reward = -10
                                elif a == 6:  # fuel
                                    if taxiloc == self.fuel_loc:
                                        new_fuel = 13
                                    else:
                                        reward = -10

                                if new_fuel <= 0:
                                    reward = -20
                                    done = True

                                newstate = self.encode(new_row, new_col, new_pass_idx, dest_idx, new_fuel)
                                P[state][a].append((1.0, newstate, reward, done))
                isd /= isd.sum()
                discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def reset(self):
        taxi_row = np.random.randint(5)
        taxi_col = np.random.randint(5)
        pass_loc = np.random.randint(4)
        dest_loc = np.random.randint(4)
        fuel = np.random.randint(6, 13)
        self.s = self.encode(taxi_row, taxi_col, pass_loc, dest_loc, fuel)
        self.lastaction = None
        return self.s

    def local_reward(self, state, action, seen_state_vars):
        if seen_state_vars == [0, 1, 2, 3, 4]:
            return self.P[state][action][0][2]
        row, col, pass_idx, dest_idx, fuel = list(self.decode(state))
        if seen_state_vars == [0, 1, 2, 4]:
            if fuel == 0:
                return - 20
            if pass_idx < 4 and action == 4 and ([row, col] == self.locs[pass_idx]):
                return 10
            if action == 6 and [row, col] == self.fuel_loc:
                return -1
            if action in [4, 5, 6]:
                return -10
            else:
                return -1
        if seen_state_vars == [0, 1, 3, 4]:
            if fuel == 0:
                return - 20
            if pass_idx == 4 and action == 5 and ([row, col] == self.locs[dest_idx]):
                return 10
            if action == 6 and [row, col] == self.fuel_loc:
                return -1
            if action in [4, 5, 6]:
                return -10
            else:
                return -1
        if seen_state_vars == [0, 1, 2]:
            if pass_idx < 4 and action == 4 and ([row, col] == self.locs[pass_idx]):
                return 10
            elif action in [4, 5, 6]:
                return -10
            else:
                return -1
        if seen_state_vars == [0, 1, 3]:
            if pass_idx == 4 and action == 5 and ([row, col] == self.locs[dest_idx]):
                return 10
            elif action in [4, 5, 6]:
                return -10
            else:
                return -1
        else:
            if action in [4, 5, 6]:
                return -10
            else:
                return -1

    def encode(self, taxirow, taxicol, passloc, destidx, fuel):
        # (5) 5, 5, 5, 3, 14
        i = taxirow
        i *= 5
        i += taxicol
        i *= 5
        i += passloc
        i *= 4
        i += destidx
        i *= 14
        i += fuel
        return i

    def decode(self, i):
        out = []
        out.append(i % 14)
        i = i // 14
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 14
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, passidx, destidx, fuelidx, fuel = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if passidx < 4:
            out[1 + taxirow][2 * taxicol + 1] = utils.colorize(out[1 + taxirow][2 * taxicol + 1], 'yellow',
                                                               highlight=True)
            pi, pj = self.locs[passidx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + taxirow][2 * taxicol + 1] = utils.colorize(ul(out[1 + taxirow][2 * taxicol + 1]), 'green',
                                                               highlight=True)

        di, dj = self.locs[destidx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile
