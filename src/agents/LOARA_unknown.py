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

    def decay(self, decay_rate):
        self.ALPHA *= decay_rate

    def update(self, action, reward):
        self.Q_table[action] += self.ALPHA * (reward - self.Q_table[action])
        self.action_visits[action] += 1


class LOARA_UK_Agent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'LOARA_unknown'
        self.params = params

        self.state_decodings = self.sweep_state_decodings()

        self.bandit_list = []
        self.state_bandit_map, self.abs_state_bandit_map = self.init_bandits()
        self.next_bandit, self.next_action = None, None

        self.sb_visits = np.zeros([self.observation_space, len(self.params.sub_spaces)])
        # self.learned_bandits = genfromtxt('helpers/learned_bandits_{}.csv'.format(str(env)), delimiter=',')

    def init_bandits(self):
        state_bandit_map = {}
        abs_state_bandit_map = [{} for ss in self.params.sub_spaces]
        for s in range(self.observation_space):
            s_vars = self.state_decodings[s]
            bandits = []
            for ia, abs_vars in enumerate(self.params.sub_spaces):
                abs_state = self.encode_abs_state(s_vars, abs_vars)
                if abs_state not in abs_state_bandit_map[ia]:
                    abs_state_bandit_map[ia][abs_state] = Bandit(self.action_space, self.params.ALPHA,
                                                                 self.params.DISCOUNT)
                    self.bandit_list.append(abs_state_bandit_map[ia][abs_state])
                    bandits.append(abs_state_bandit_map[ia][abs_state])
                else:
                    bandits.append(abs_state_bandit_map[ia][abs_state])
            state_bandit_map[s] = bandits
        return state_bandit_map, abs_state_bandit_map

    def sweep_state_decodings(self):
        st_vars_lookup = []
        for s in range(self.observation_space):
            st_vars_lookup.append(list(self.env.decode(s)))
        return st_vars_lookup

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
            for b in self.bandit_list:
                b.decay(decay_rate)
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def e_greedy_bandit(self, state):
        if random.uniform(0, 1) < self.params.EPSILON:
            self.next_action = self.state_bandit_map[state][-1].random_action()
            return np.random.randint(len(self.params.sub_spaces) - 1), max(self.Q_table[state])
        bandit_index = int(np.argmax(self.Q_table[state]))
        self.next_action = self.state_bandit_map[state][bandit_index].random_action()
        return bandit_index, max(self.Q_table[state])

    def smart_bandit_choice(self, state, eps=None):
        values = [max(b.Q_table) for ib, b in enumerate(self.state_bandit_map[state])]
        if np.random.uniform(0, 1) < (eps if eps else self.params.EPSILON):
            self.next_action = self.state_bandit_map[state][-1].random_action()
            return np.random.randint(len(self.params.sub_spaces)), max(values)
        bandit_index = int(np.argmax(values))
        self.next_action = self.state_bandit_map[state][bandit_index].greedy_action()
        return bandit_index, max(values)

    def e_greedy_bandit_action(self, state):
        if self.next_bandit is None:
            bandit_index = np.argmax([max(b.Q_table) for b in self.state_bandit_map[state]])
        else:
            bandit_index = self.next_bandit
        if self.next_action is None:
            if random.uniform(0, 1) < self.params.EPSILON:
                action = self.state_bandit_map[state][bandit_index].random_action()
            else:
                action = self.state_bandit_map[state][bandit_index].greedy_action()
        else:
            action = self.next_action
        self.next_bandit = None
        self.next_action = None
        return bandit_index, action

    def update_LIA(self, state, bandit_index, action, reward, next_state, done):
        self.next_bandit, val = self.smart_bandit_choice(next_state)
        next_val = (not done) * val
        for ib, b in enumerate(self.state_bandit_map[state]):
            if ib == len(self.params.sub_spaces) - 1:
                self.state_bandit_map[state][-1].update(action, reward + self.params.DISCOUNT * max(
                    self.state_bandit_map[next_state][-1].Q_table))
            elif set(self.params.sub_spaces[ib]).issubset(set(self.params.sub_spaces[bandit_index])):
                self.state_bandit_map[state][bandit_index].update(action, reward + self.params.DISCOUNT * next_val)

        # self.state_bandit_map[state][bandit_index].update(action, reward + self.params.DISCOUNT * next_val)
        # if bandit_index != len(self.params.sub_spaces) - 1:
        #     self.state_bandit_map[state][-1].update(action, reward + self.params.DISCOUNT * max(self.state_bandit_map[next_state][-1].Q_table))

    def do_step(self):
        state = self.current_state
        bandit_index, action = self.e_greedy_bandit_action(state)
        next_state, reward, done = self.step(action)
        self.sb_visits[state][bandit_index] += 1
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update_LIA(state, bandit_index, action, reward, next_state, done)
        self.steps += 1
        self.next_action = None
        self.next_bandit = None
        return reward, done
