import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+---------+",
    "|B: : | :M|",
    "| : : | : |",
    "| : : : : |",
    "| : | : : |",
    "| :C| : :A|",
    "+---------+",
]

BIGMAP = [
    "+-------------+",
    "|B: : | :M: : |",
    "| : : | : : : |",
    "| : : : : : : |",
    "| : : : : : : |",
    "| : : | : : : |",
    "| : : | : : : |",
    "| :C: | : : :A|",
    "+-------------+",
]

class CoffeeMailEnv(discrete.DiscreteEnv):
    """
    The Coffee Mail problem

    Description:
    A robot exists in a 5x5 office grid world and is tasked with getting coffee and/or mail for workers A and/or B.
    The locations of the coffee machine, mail box and workers A and B are fixed. An episode ends after __ time steps.

    Observations: 
    There are 1600 states as there are 25 locations for the robot, and binary variables for whether the robot has
    coffee, the robot has mail, worker A wants coffee, worker B wants coffee, worker A wants mail and worker B wants mail.
        
    Actions:
    There are 7 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: take
    - 5: give
    - 6: do nothing
    
    Rewards: 
    There is a reward of -1 for every time step if worker A or B wants coffee and additional -1 reward each time step if
    both want coffee. There is a reward of -0.5 for every time step if worker A or B wants mail and additional -0.5 if
    both want mail. There is a reward of 20 for delivering coffee to a worker who wants it. There is a reward of 20 for
    delivering mail to a worker who wants it.
    

    state space is represented by:
        (robot_x, robot_y, robot_hascoffee, robot_hasmail, A_wantscoffee, B_wantscoffee, A_wantsmail, B_wantsmail)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')
        self.c_loc = (4, 1)
        self.m_loc = (0, 4)
        self.A_loc = (4, 4)
        self.B_loc = (0, 0)
        self.num_rows = 5
        self.num_columns = 5

        # ---IF BIG
        self.desc = np.asarray(BIGMAP, dtype='c')
        self.c_loc = (6, 1)
        self.A_loc = (6, 6)
        self.num_rows = 7
        self.num_columns = 7

        self.state_desc = ['x', 'y', 'rc', 'rm', 'ac', 'bc', 'am', 'bm']

        num_states = self.num_rows * self.num_columns * 2**6
        max_row = self.num_rows - 1
        max_col = self.num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 7
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for x in range(self.num_rows):
            for y in range(self.num_columns):
                for rc in range(2):  # robot has coffee
                    for rm in range(2):  # robot has mail
                        for ac in range(2):  # A wants coffee
                            for bc in range(2):  # B wants coffee
                                for am in range(2):  # A wants mail
                                    for bm in range(2):  # B wants mail
                                        state = self.encode(x, y, rc, rm, ac, bc, am, bm)
                                        initial_state_distrib[state] += 1
                                        for action in range(num_actions):
                                            # defaults
                                            new_x, new_y, new_rc, new_rm, new_ac, new_bc, new_am, new_bm = x, y, rc, rm, ac, bc, am, bm
                                            reward = 0  # default reward when there is no pickup/dropoff
                                            done = False
                                            r_loc = (x, y)

                                            if ac and bc:
                                                reward += -2
                                            elif ac or bc:
                                                reward += -1
                                            if am and bm:
                                                reward += -1
                                            elif am or bm:
                                                reward += -0.5

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
                                                elif r_loc == self.m_loc:
                                                    if not rm:
                                                        new_rm = 1
                                            elif action == 5:  # give
                                                if r_loc == self.A_loc:
                                                    if ac and rc:
                                                        reward += 20
                                                        new_ac = 0
                                                        new_rc = 0
                                                    if am and rm:
                                                        reward += 20
                                                        new_am = 0
                                                        new_rm = 0

                                                if r_loc == self.B_loc:
                                                    if bc and rc:
                                                        reward += 20
                                                        new_bc = 0
                                                        new_rc = 0
                                                    if bm and rm:
                                                        reward += 20
                                                        new_bm = 0
                                                        new_rm = 0
                                            # elif action == 6:  # do nothing

                                            if new_ac == new_bc == new_am == new_bm == 0:
                                                done = True

                                            new_state = self.encode(
                                                new_x, new_y, new_rc, new_rm, new_ac, new_bc, new_am, new_bm)
                                            P[state][action].append(
                                                (1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def reset(self):
        x = np.random.randint(0, self.num_rows)
        y = np.random.randint(0, self.num_columns)
        rc = 0
        rm = 0
        ac = np.random.randint(0, 2)
        bc = np.random.randint(0, 2)
        am = np.random.randint(0, 2)
        bm = np.random.randint(0, 2)
        self.s = self.encode(x, y, rc, rm, ac, bc, am, bm)
        self.lastaction = None
        return self.s


    def encode(self, x, y, rc, rm, ac, bc, am, bm):
        # (5) 5, 2, 2, 2, 2, 2, 2
        i = x
        i *= self.num_columns
        i += y
        i *= 2
        i += rc
        i *= 2
        i += rm
        i *= 2
        i += ac
        i *= 2
        i += bc
        i *= 2
        i += am
        i *= 2
        i += bm
        return i

    def decode(self, i):
        out = []
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % self.num_columns)
        i = i // self.num_columns
        out.append(i)
        assert 0 <= i <= self.num_columns
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x): return "_" if x == " " else x
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
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
