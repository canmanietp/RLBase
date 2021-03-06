import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


class NoisyTaxiEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.

    Observations: 
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations. 
    
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
        
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rewards: 
    There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
    

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    

    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')

        self.locs = locs = [(0,0), (0,4), (4,0), (4,3)]

        self.noise_options = 5
        num_states = 500 * self.noise_options
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        for noise in range(self.noise_options):
                            state = self.encode(row, col, pass_idx, dest_idx, noise)
                            if pass_idx < 4 and pass_idx != dest_idx:
                                initial_state_distrib[state] += 1
                            for action in range(num_actions):
                                # defaults
                                new_row, new_col, new_pass_idx = row, col, pass_idx
                                reward = -1 # default reward when there is no pickup/dropoff
                                done = False
                                taxi_loc = (row, col)

                                if action == 0:
                                    new_row = min(row + 1, max_row)
                                elif action == 1:
                                    new_row = max(row - 1, 0)
                                if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                    new_col = min(col + 1, max_col)
                                elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                    new_col = max(col - 1, 0)
                                elif action == 4:  # pickup
                                    if pass_idx < 4 and taxi_loc == locs[pass_idx]:
                                        new_pass_idx = 4
                                    else: # passenger not at location
                                        reward = -10
                                elif action == 5:  # dropoff
                                    if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
                                        new_pass_idx = dest_idx
                                        done = True
                                        reward = 20
                                    elif (taxi_loc in locs) and pass_idx == 4:
                                        new_pass_idx = locs.index(taxi_loc)
                                    else: # dropoff at wrong location
                                        reward = -10
                                new_state = self.encode(
                                    new_row, new_col, new_pass_idx, dest_idx, np.random.randint(0, self.noise_options))
                                P[state][action].append(
                                    (1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def local_reward(self, state, action, seen_state_vars):
        if seen_state_vars == [0, 1, 2, 3]:
            return self.P[state][action][0][2]
        row, col, pass_idx, dest_idx = list(self.decode(state))
        if seen_state_vars == [0, 1, 2]:
            if pass_idx < 4 and action == 4 and ([row, col] == self.locs[pass_idx]):
                return 10
            elif action in [4, 5]:
                return -10
            else:
                return -1
        if seen_state_vars == [0, 1, 3]:
            if pass_idx == 4 and action == 5 and ([row, col] == self.locs[dest_idx]):
                return 10
            elif action in [4, 5]:
                return -10
            else:
                return -1
        else:
            if action in [4, 5]:
                return -10
            else:
                return -1

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx, noise):
        # (5) 5, 5, 4, noise_options
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        i *= self.noise_options
        i += noise
        return i

    def decode(self, i):
        out = []
        out.append(i % self.noise_options)
        i = i // self.noise_options
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < max(5, self.noise_options)
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
