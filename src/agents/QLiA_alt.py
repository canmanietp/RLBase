import random

from agents.Q import QAgent
from agents.Q import QMiniAgent
from learning_parameters import DiscreteParameters
import numpy as np
import copy


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

    def update(self, action, reward, next_value, done):
        self.Q_table[action] += self.ALPHA * (reward + (not done) * self.DISCOUNT * next_value - self.Q_table[action])
        # (1 / (self.action_visits[action] + 1))
        self.action_visits[action] += 1


class QLiA_altAgent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'QLiA_alt'
        self.params = params

        self.action_space = len(params.sub_spaces)
        self.Q_table = np.zeros([self.observation_space, self.action_space])
        self.state_decodings = self.sweep_state_decodings()

        self.state_bandit_map = {}  # state: bandit
        self.next_bandit, self.next_action = None, None
        # self.init_bandits()
        self.bandits = self.init_bandits()

    def init_bandits(self):
        bandits = []
        bandit_count = 0
        for s in range(self.observation_space):
            if s not in self.state_bandit_map:
                state_vars = self.state_decodings[s]
                if state_vars[1] == 1 and state_vars[2] == 4 and state_vars[3] == 0:
                    x = 0
                # elif state_vars[0] == 0 and state_vars[2] == 4:
                #     x = 1
                # elif state_vars[2] == 4:
                #     x = 2
                # elif (state_vars[0] == 3 and state_vars[1] == 2) or (state_vars[0] == 3 and state_vars[1] == 1):
                #     x = 3
                else:
                    x = 4
                abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[x])
                for _s in range(self.observation_space):
                    _state_vars = self.state_decodings[_s]
                    abs_s = self.encode_abs_state(_state_vars, self.params.sub_spaces[x])
                    if abs_s == abs_state:
                        if _s in self.state_bandit_map:
                            self.state_bandit_map[s] = self.state_bandit_map[_s]
                            break
                if s not in self.state_bandit_map:
                    self.state_bandit_map[s] = Bandit(self.env.action_space.n, self.params.ALPHA, self.params.DISCOUNT)
                    bandit_count += 1
                    print("bandy", bandit_count)
        return bandits

    # def init_bandits(self):
    #     for s in range(self.observation_space):
    #         state_vars = self.state_decodings[s]
    #         bandits = [[] for b in self.params.sub_spaces]
    #         for isx, ss, in enumerate(self.params.sub_spaces):
    #             abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[isx])
    #             for _s in range(self.observation_space):
    #                 _state_vars = self.state_decodings[_s]
    #                 abs_s = self.encode_abs_state(_state_vars, self.params.sub_spaces[isx])
    #                 if abs_s == abs_state:
    #                     if _s in self.state_bandit_map:
    #                         bandits[isx] = self.state_bandit_map[_s][isx]
    #                         break
    #             if not bandits[isx]:
    #                 bandits[isx] = Bandit(self.env.action_space.n, 0.3, 0.95)
    #         self.state_bandit_map[s] = bandits

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

    def e_greedy_bandit_action(self, state):
        # self.add_state_to_bandit_map(state)

        if self.next_bandit is not None:
            return self.next_bandit, self.next_action
        bandit = None
        if random.uniform(0, 1) < self.params.EPSILON:
            # bandit = self.random_action()
            action = self.state_bandit_map[state].random_action()

        else:
            # bandit = self.greedy_action(state)
            action = self.state_bandit_map[state].greedy_action()
        return bandit, action

    def update_LIA(self, state, bandit, action, reward, next_state, done):
        # self.next_bandit, self.next_action = None, None
        # self.next_bandit, self.next_action = self.e_greedy_bandit_action(state)
        # self.add_state_to_bandit_map(next_state)
        next_val = max(self.state_bandit_map[next_state].Q_table)if not done else 0  # self.state_bandit_map[next_state][self.next_bandit].Q_table[self.next_action]
        self.state_bandit_map[state].update(action, reward, next_val, done)
        # print(state_vars, self.true_agent.Q_table[state], self.state_bandit_map[state].Q_table, action)
        # print(state_vars, self.state_bandit_map[state].Q_table, self.state_bandit_map[next_state].Q_table)
        # self.update(state, bandit, reward, next_state, done)

    def do_step(self):
        state = self.current_state
        bandit, action = self.e_greedy_bandit_action(state)
        next_state, reward, done = self.step(action)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update_LIA(state, bandit, action, reward, next_state, done)
        self.current_state = next_state
        self.steps += 1
        return reward, done
