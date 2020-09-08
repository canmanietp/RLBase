import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np
import copy

Size1_MAP = [
    "+-------+",
    "| : : :1|",
    "| : : : |",
    "| :0: : |",
    "+-------+",
]

Size2_MAP = [
    "+---------------+",
    "| : : :1: : : : |",
    "| : : : : : : : |",
    "| :0: : : : : :3|",
    "| : : : : : : : |",
    "| : : : :2: : : |",
    "+---------------+",
]

Size3_MAP = [
    "+---------------------+",
    "| : : :1: : : : : :5: |",
    "| : : : : : : : : : : |",
    "| :0: : : : : :3: : : |",
    "| : : : : : : : : : : |",
    "| : : : :2: : : : :4: |",
    "+---------------------+",
]


class WarehouseEnv(discrete.DiscreteEnv):
    """

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, size):
        self.locs = []
        self.num_products = num_products = size * 2

        # For exponential scaling
        # size ** 2
        # fill in locations of each product
        # for i in range(num_products):
        #     self.locs.append((int(i / size) * size + 1, (i % size) * size + 1))

        if size == 1:
            self.locs = [(2, 1), (0, 3)]
        elif size == 2:
            self.locs = [(2, 1), (0, 3), (4, 4), (2, 7)]
        elif size == 3:
            self.locs = [(2, 1), (0, 3), (4, 4), (2, 7), (4, 9), (0, 9)]

        self.locs.append((0, 0))  # start position
        # maximum number of each product that can be requested (hard coded)
        self.max_order = max_order = 2

        # all possible number of requests for each product times possible location at each product + start position
        num_states = (max_order + 1) ** num_products * (num_products + 1)
        initial_state_distrib = np.zeros(num_states)
        initial_state_distrib[num_states - 1] = 1
        num_actions = self.num_products + 1  # 1 for go back to start
        P = {state: {action: [] for action in range(num_actions)} for state in range(num_states)}

        for product_request in range((max_order + 1) ** num_products):  # +1 for none ordered
            for location in range(len(self.locs)):
                req = list(self.decode_requests(product_request))
                raw_state = [location]
                raw_state.extend(req)
                state = self.encode(*raw_state)

                for action in range(num_actions):
                    new_req = copy.copy(req)
                    new_loc = action
                    done = False

                    reward = -1 * (abs(self.locs[location][0] - self.locs[new_loc][0]) + abs(
                    self.locs[location][1] - self.locs[new_loc][1])) - 1

                    if action < len(self.locs) - 1:
                        if req[action] > 0:
                            new_req[action] -= 1
                    else:
                        if all(nr == 0 for nr in new_req):
                            done = True
                        else:
                            reward -= 10

                    new_raw_state = [new_loc]
                    new_raw_state.extend(new_req)
                    new_state = self.encode(*new_raw_state)
                    P[state][action].append((1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(self, num_states, num_actions, P, initial_state_distrib)

    # Override usual reset function
    # def reset(self):
    #     loc = np.random.randint(0, len(self.locs))
    #     reqs = []
    #     for r in range(self.num_products):
    #         reqs.append(np.random.randint(0, self.max_order + 1))
    #     raw_state = [loc]
    #     raw_state.extend(reqs)
    #     self.s = self.encode(*raw_state)
    #     self.lastaction = None
    #     return self.s

    def encode(self, location, *args):
        # (location) requests of product A, requests of product B, etc.
        i = location
        for prod in args:
            i *= (self.max_order + 1)
            i += prod
        return i

    def decode(self, i):
        # (location), max_order + 1, max_order + 1, ...
        out = []
        for p in range(self.num_products):
            out.append(i % (self.max_order + 1))
            i = i // (self.max_order + 1)
        out.append(i)
        assert 0 <= i < self.num_products + 1
        return reversed(out)

    def decode_requests(self, i):
        # (max_order + 1), max_order + 1, ...
        out = []
        for p in range(self.num_products):
            if p != 0:
                out.append(i % (self.max_order + 1))
                i = i // (self.max_order + 1)
        out.append(i)
        assert 0 <= i <= self.max_order
        return reversed(out)

# size = 2
# blah = WarehouseEnv(size)
# #
# for i in range(1):
#     s = blah.reset()
#     done = False
#     while not done:
#         a = np.random.randint(size ** 2 + 1)
#         ns, reward, done, _ = blah.step(a)
#         print(list(blah.decode(s)), a, list(blah.decode(ns)), reward, done)
#         s = ns
