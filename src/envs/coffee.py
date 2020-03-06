import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+---------+",
    "|B: : | : |",
    "| : : | : |",
    "| : : : : |",
    "| : | : : |",
    "| :C| : :A|",
    "+---------+",
]


class CoffeeEnv(discrete.DiscreteEnv):
    """
    The Coffee problem

    Description:
    A robot exists in a 5x5 office grid world and is tasked with getting coffee for workers A and/or B.
    The locations of the coffee machine and workers A and B are fixed. An episode ends after __ time steps.

    Observations: 
    There are 200 states as there are 25 locations for the robot, and binary variables for whether the robot has
    coffee, the robot has mail, worker A wants coffee, worker B wants coffee, worker A wants mail and worker B wants mail.
        
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: take
    - 5: give
    - 6: do nothing
    
    Rewards: 
    There is a reward of -1 for every time step if worker A or B wants coffee and additional -1 reward each time step if
    both want coffee. There is a reward of 20 for delivering coffee to a worker who wants it.
    

    state space is represented by:
        (robot_x, robot_y, robot_hascoffee, A_wantscoffee, B_wantscoffee)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')

        self.c_loc = (4, 1)
        self.A_loc = (4, 4)
        self.B_loc = (0, 0)

        num_states = 200
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 7
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for x in range(num_rows):
            for y in range(num_columns):
                for rc in range(2):  # robot has coffee
                    for ac in range(2):  # A wants coffee
                        for bc in range(2):  # B wants coffee
                            state = self.encode(x, y, rc, ac, bc)
                            initial_state_distrib[state] += 1
                            for action in range(num_actions):
                                # defaults
                                new_x, new_y, new_rc, new_ac, new_bc = x, y, rc, ac, bc
                                reward = 0  # default reward when there is no pickup/dropoff
                                done = False
                                r_loc = (x, y)

                                if ac and bc:
                                    reward += -2
                                elif ac or bc:
                                    reward += -1

                                if action == 0 and self.desc[x, y] != b"_":  # down
                                    new_x = min(x + 1, max_row)
                                elif action == 1 and self.desc[x, y - 1] != b"_":  # up
                                    new_x = max(x - 1, 0)
                                if action == 2 and self.desc[1 + x, 2 * y + 2] == b":":  # right
                                    new_y = min(y + 1, max_col)
                                elif action == 3 and self.desc[1 + x, 2 * y] == b":":  # left
                                    new_y = max(y - 1, 0)
                                elif action == 4:  # take
                                    if r_loc == self.c_loc:
                                        if not rc:
                                            new_rc = 1
                                elif action == 5:  # give
                                    if r_loc == self.A_loc:
                                        if ac and rc:
                                            reward += 20
                                            new_ac = 0
                                            new_rc = 0
                                    if r_loc == self.B_loc:
                                        if bc and rc:
                                            reward += 20
                                            new_bc = 0
                                            new_rc = 0
                                # elif action == 6:  # do nothing

                                if new_ac == new_bc == 0:
                                    done = True

                                new_state = self.encode(
                                    new_x, new_y, new_rc, new_ac, new_bc)
                                P[state][action].append(
                                    (1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def reset(self):
        x = np.random.randint(0, 5)
        y = np.random.randint(0, 5)
        rc = 0
        ac = np.random.randint(0, 2)
        bc = np.random.randint(0, 2)
        self.s = self.encode(x, y, rc, ac, bc)
        self.lastaction = None
        return self.s

    def encode(self, x, y, rc, ac, bc):
        # (5) 5, 2, 2, 2
        i = x
        i *= 5
        i += y
        i *= 2
        i += rc
        i *= 2
        i += ac
        i *= 2
        i += bc
        return i

    def decode(self, i):
        out = []
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
