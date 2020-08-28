import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np


class FourStateEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):

        num_states = 4
        num_x = 2
        num_y = 2
        initial_state_distrib = np.zeros(num_states)
        num_actions = 2
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for x in range(num_x):
            for y in range(num_y):
                state = self.encode(x, y)
                for action in range(num_actions):
                    # defaults
                    new_x, new_y = x, y
                    done = False
                    reward = 0

                    if action == 0:  # solid arrow
                        if (new_x, new_y) == (0, 0):
                            reward = 0
                            new_x = 1
                            new_y = 1
                        elif (new_x, new_y) == (0, 1):
                            reward = 0
                            new_x = 1
                            new_y = 1
                        elif (new_x, new_y) == (1, 0):
                            reward = 0
                            done = True
                        elif (new_x, new_y) == (1, 1):
                            reward = 3
                            done = True
                    elif action == 1:  # dashed arrow
                        if (new_x, new_y) == (0, 0):
                            reward = 1
                            new_x = 1
                            new_y = 0
                        elif (new_x, new_y) == (0, 1):
                            reward = 1
                            new_x = 1
                            new_y = 0
                        elif (new_x, new_y) == (1, 0):
                            reward = 1
                            done = True
                        elif (new_x, new_y) == (1, 1):
                            reward = 3
                            done = True

                    new_state = self.encode(new_x, new_y)
                    P[state][action].append((1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def reset(self):
        x = 0  # np.random.randint(0, 2)
        y = np.random.randint(0, 2)
        self.s = self.encode(x, y)
        self.lastaction = None
        return self.s

    def encode(self, x, y):
        # (2) 2
        i = x
        i *= 2
        i += y
        return i

    def decode(self, i):
        out = [i % 2]
        i = i // 2
        out.append(i)
        assert 0 <= i < 2
        return reversed(out)
