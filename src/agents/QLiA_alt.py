import math
import random
import time

from agents.Q import QAgent
import numpy as np
import copy

from numpy import genfromtxt

class Bandit:
    bandit_count = 0

    def __init__(self, action_space, alpha, discount):
        self.action_space = action_space
        self.Q_table = np.zeros(action_space)
        self.action_visits = np.zeros(action_space)
        Bandit.bandit_count += 1
        self.ALPHA = alpha
        self.DISCOUNT = discount

    def random_action(self):
        return np.random.randint(self.action_space)

    def greedy_action(self):
        qv = self.Q_table
        return np.random.choice(np.flatnonzero(qv == max(qv)))

    def ucb1_action(self):
        action = self.random_action()
        if np.sum(self.action_visits) > 1:
            qv = self.Q_table + np.sqrt(np.divide(2 * math.log(np.sum(self.action_visits)), self.action_visits))
            action = np.random.choice(np.flatnonzero(qv == max(qv)))
        return action

    def update(self, action, reward):
        self.Q_table[action] += self.ALPHA * (reward - self.Q_table[action])
        self.action_visits[action] += 1


class QLiA_altAgent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'QLiA_alt'
        self.params = params

        self.action_space = len(params.sub_spaces)
        self.Q_table = np.zeros([self.observation_space, self.action_space])  # self.env.action_space.n])  #
        self.sabs_visits = np.zeros([self.observation_space, self.action_space])
        self.state_decodings = self.sweep_state_decodings()

        self.state_bandit_map = {}  # state: bandit
        self.state_abs_state = [[[] for b in self.params.sub_spaces] for s in range(self.observation_space)]
        self.next_bandit, self.next_action = None, None

        # self.learned_bandits = genfromtxt('helpers/learned_bandits_{}.csv'.format(str(env)), delimiter=',')

        print("Initializing bandits ...")
        self.bandits = self.init_bandits()
        print("Done.")

    def init_bandits(self):
        return_bandits = []
        for s in range(self.observation_space):
            state_vars = self.state_decodings[s]
            bandits = [[] for b in self.params.sub_spaces]
            for isx, ss, in enumerate(self.params.sub_spaces):
                next_states = []
                if isx != len(self.params.sub_spaces) - 1:
                    abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[isx])
                    for _s in range(self.observation_space):
                        _state_vars = self.state_decodings[_s]
                        abs_s = self.encode_abs_state(_state_vars, self.params.sub_spaces[isx])
                        if abs_s == abs_state:
                            if _s in self.state_bandit_map:
                                bandits[isx] = self.state_bandit_map[_s][isx]
                                next_states.append(_s)
                    self.state_abs_state[s][isx] = next_states
                else:
                    self.state_abs_state[s][isx] = [s]
                if not bandits[isx]:
                    bandits[isx] = Bandit(self.env.action_space.n, self.params.ALPHA, self.params.DISCOUNT)
                    return_bandits.append(bandits[isx])
            self.state_bandit_map[s] = bandits
        return return_bandits

    def add_state_to_bandit_map(self, state):
        if state not in self.state_bandit_map:
            state_vars = self.state_decodings[state]
            bandits = [[] for b in self.params.sub_spaces]
            for isx, ss in enumerate(self.params.sub_spaces):
                abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[isx])
                for _s in range(self.observation_space):
                    _state_vars = self.state_decodings[_s]
                    _abs_state = self.encode_abs_state(_state_vars, self.params.sub_spaces[isx])
                    if _abs_state == abs_state:
                        if _s in self.state_bandit_map:
                            bandits[isx] = self.state_bandit_map[_s][isx]
                            break
                    if not bandits[isx]:
                        bandits[isx] = Bandit(self.env.action_space.n, self.params.ALPHA, self.params.DISCOUNT)
            self.state_bandit_map[state] = bandits

    def sweep_state_decodings(self):
        st_vars_lookup = []
        for s in range(self.observation_space):
            st_vars_lookup.append(list(self.env.decode(s)))
        return st_vars_lookup

    def state_to_abs_to_states(self, state, ab_index):
        if ab_index == len(self.params.sub_spaces) - 1:
            return [state]
        if not self.state_abs_state[state][ab_index]:
            next_states = []
            state_vars = self.state_decodings[state]
            abs_s = self.encode_abs_state(state_vars, self.params.sub_spaces[ab_index])
            for _s in range(self.observation_space):
                _s_vars = self.state_decodings[_s]
                _abs_s = self.encode_abs_state(_s_vars, self.params.sub_spaces[ab_index])
                if _abs_s == abs_s:
                    next_states.append(_s)
            self.state_abs_state[state][ab_index] = next_states
            return next_states
        else:
            return self.state_abs_state[state][ab_index]

    def encode_abs_state(self, state, abstraction):
        abs_state = [state[k] for k in abstraction]
        var_size = copy.copy([self.params.size_state_vars[k] for k in abstraction])
        var_size.pop(0)
        encoded_state = 0

        for e in range(len(abs_state) - 1):
            encoded_state += abs_state[e] * np.prod(var_size)
            var_size.pop(0)

        encoded_state += abs_state[-1]
        return encoded_state

    def decay(self, decay_rate):
        if self.params.ALPHA > self.params.ALPHA_MIN:
            self.params.ALPHA *= decay_rate
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

        for b in self.bandits:
            if b.ALPHA > self.params.ALPHA_MIN:
                b.ALPHA *= decay_rate

    def heuristic_bandit_choice(self, state):
        state_vars = self.state_decodings[state]
        # # TAXI
        # 0 if (state_vars[0] == 4 and state_vars[1] == 1 or state_vars[0] == 4 and state_vars[1] == 2) else \
        #     1 if (state_vars[0] == 3 and state_vars[2] == 4 and state_vars[3] == 0) else \
        #     2 if (state_vars[1] == 4 and state_vars[2] == 4 and state_vars[3] != 1) else \
        return 0 if (state_vars[2] < 4) else 1
        # COFFEE
        # return 0 if (state_vars[2] == 0) else 1
        # COFFEE MAIL
        # return 0 if (state_vars[3] == 1 or state_vars[4] == 1) else 1 if (state_vars[5] == 1 or state_vars[6] == 1) else 2

    def smart_bandit_choice(self, state, eps=None):
        values = [max(b.Q_table) for b in self.state_bandit_map[state]]
        if np.random.uniform(0, 1) < (eps if eps else self.params.EPSILON):
            return self.random_action(), max(values)
        return np.argmax(values), max(values)

    def e_greedy_bandit_action(self, state):
        # self.add_state_to_bandit_map(state)
        # self.update(state, bandit, reward, next_state, done)
        if self.next_bandit is not None:
            bandit = self.next_bandit
        else:
            bandit = self.random_action()
        if random.uniform(0, 1) < self.params.EPSILON:
            action = self.state_bandit_map[state][bandit].random_action()
        else:
            action = self.state_bandit_map[state][bandit].greedy_action()
        return bandit, action

    def update_LIA(self, state, bandit, action, reward, next_state, done):
        # self.add_state_to_bandit_map(next_state)
        self.next_bandit, val = self.smart_bandit_choice(next_state)  # self.heuristic_bandit_choice(next_state) # self.smart_bandit_choice(next_state)  # self.greedy_action(next_state) # self.heuristic_bandit_choice(next_state) #  # self.smart_bandit_choice(next_state) #
        next_val = (not done) * val  # max(self.state_bandit_map[next_state][self.next_bandit].Q_table)
        self.state_bandit_map[state][bandit].update(action, reward + self.params.DISCOUNT * next_val)
        if bandit != len(self.params.sub_spaces) - 1:
            self.state_bandit_map[state][-1].update(action, reward + self.params.DISCOUNT * next_val) # max(self.state_bandit_map[next_state][-1].Q_table))

    def do_step(self):
        state = self.current_state
        bandit, action = self.e_greedy_bandit_action(state)
        next_state, reward, done = self.step(action)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update_LIA(state, bandit, action, reward, next_state, done)
        self.current_state = next_state
        self.steps += 1
        self.sabs_visits[state][bandit] += 1
        self.sa_visits[state][action] += 1
        return reward, done
