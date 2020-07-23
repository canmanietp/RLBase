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


class LOARA_K_Agent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'LOARA_known'
        self.params = params

        self.state_decodings = self.sweep_state_decodings()

        self.state_bandit_map, self.abs_state_bandit_map, self.bandit_list = self.init_bandits()
        self.next_bandit, self.next_action = None, None
        self.bandit_list = []

    def init_bandits(self):
        state_bandit_map = {}
        abs_state_bandit_map = [{} for ss in self.params.sub_spaces]
        bandit_list = []
        for s in range(self.observation_space):
            s_vars = self.state_decodings[s]
            bandits = []
            for ia, abs_vars in enumerate(self.params.sub_spaces):
                abs_state = self.encode_abs_state(s_vars, abs_vars)
                if abs_state not in abs_state_bandit_map[ia]:
                    abs_state_bandit_map[ia][abs_state] = Bandit(self.action_space, self.params.ALPHA, self.params.DISCOUNT)
                    bandit_list.append(abs_state_bandit_map[ia][abs_state])
                    bandits.append(abs_state_bandit_map[ia][abs_state])
                else:
                    bandits.append(abs_state_bandit_map[ia][abs_state])
            state_bandit_map[s] = bandits
        return state_bandit_map, abs_state_bandit_map, bandit_list

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
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

        for b in self.bandit_list:
            if b.ALPHA > self.params.ALPHA_MIN:
                b.ALPHA *= decay_rate

    def heuristic_bandit_choice(self, state):
        state_vars = self.state_decodings[state]
        if 'TaxiFuel' in str(self.env):
            if state_vars[2] < 4 and state_vars[4] > 12: # don't have passenger and have enough fuel to get passenger and to destination
                return self.params.sub_spaces.index([0, 1, 2])
            elif state_vars[2] == 4 and state_vars[4] > 8:
                return self.params.sub_spaces.index([0, 1, 2, 3])
            else:
                return len(self.params.sub_spaces) - 1
        elif 'Taxi' in str(self.env):
            if state_vars[0] == 4 and state_vars[1] == 1 or state_vars[0] == 4 and state_vars[1] == 2:
                return self.params.sub_spaces.index([2])
            elif state_vars[0] == 3 and state_vars[2] == 4 and state_vars[3] == 0:
                return self.params.sub_spaces.index([0, 2, 3])
            elif state_vars[1] == 4 and state_vars[2] == 4 and state_vars[3] != 1:
                return self.params.sub_spaces.index([1, 2, 3])
            elif state_vars[2] < 4:
                return self.params.sub_spaces.index([0, 1, 2])
            else:
                return len(self.params.sub_spaces) - 1
        elif 'CoffeeMail' in str(self.env):
            if state_vars[3] == 1 or state_vars[4] == 1:
                return self.params.sub_spaces.index([0, 1, 2, 3, 4])
            elif state_vars[5] == 1 or state_vars[6] == 1:
                return self.params.sub_spaces.index([0, 1, 2, 5, 6])
            else:
                return len(self.params.sub_spaces) - 1
        elif 'Coffee' in str(self.env):
            if state_vars[2] == 0:
                return self.params.sub_spaces.index([0, 1, 2])
            else:
                return len(self.params.sub_spaces) - 1
        else:
            print("ERROR: UNKNOWN HEURISTIC FOR CHOOSING ABSTRACTION")
            quit()

    def e_greedy_bandit_action(self, state):
        if self.next_bandit is None:
            bandit_index = np.random.randint(len(self.params.sub_spaces))
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
        self.next_bandit = self.heuristic_bandit_choice(next_state)
        val = max(self.state_bandit_map[next_state][self.next_bandit].Q_table)
        next_val = (not done) * val
        self.state_bandit_map[state][bandit_index].update(action, reward + self.params.DISCOUNT * next_val)
        if bandit_index != len(self.params.sub_spaces) - 1:
            self.state_bandit_map[state][-1].update(action, reward + self.params.DISCOUNT * max(self.state_bandit_map[next_state][self.next_bandit].Q_table))  # next_val)

    def do_step(self):
        state = self.current_state
        bandit_index, action = self.e_greedy_bandit_action(state)
        next_state, reward, done = self.step(action)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update_LIA(state, bandit_index, action, reward, next_state, done)
        self.current_state = next_state
        self.steps += 1
        return reward, done