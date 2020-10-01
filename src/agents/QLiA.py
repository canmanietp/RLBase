import random
import time

from agents.Q import QAgent
from agents.Q import QMiniAgent
from learning_parameters import DiscreteParameters
import numpy as np
import copy


class QLiAAgent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'QLiA'
        self.sub_agents = []
        self.params = copy.copy(params)

        for ab in params.sub_spaces:
            ss = 1
            for var in ab:
                ss *= params.size_state_vars[var]

            ab_params = copy.copy(self.params)
            ab_params.EPSILON = params.PHI
            ab_params.EPSILON_MIN = params.PHI_MIN
            self.sub_agents.append(QMiniAgent(self.env, ab_params, self.observation_space, self.env.action_space.n))

        self.action_space = len(params.sub_spaces)
        self.Q_table = np.zeros([self.observation_space, self.action_space])

        self.state_decodings = self.sweep_state_decodings()
        self.next_abstraction = None

    def sweep_state_decodings(self):
        st_vars_lookup = []
        for s in range(self.observation_space):
            st_vars_lookup.append(list(self.env.decode(s)))
        return st_vars_lookup

    def encode_abs_state(self, state, abstraction):
        new_state = copy.copy(state)

        for iv, var in enumerate(new_state):
            if iv not in abstraction:
                new_state[iv] = 1

        return self.env.encode(*new_state)
        #
        # abs_state = [state[k] for k in abstraction]
        # var_size = copy.copy([self.params.size_state_vars[k] for k in abstraction])
        # var_size.pop(0)
        # encoded_state = 0
        #
        # for e in range(len(abs_state) - 1):
        #     encoded_state += abs_state[e] * np.prod(var_size)
        #     var_size.pop(0)
        #
        # encoded_state += abs_state[-1]
        # return encoded_state

    def decay(self, decay_rate):
        if self.params.ALPHA > self.params.ALPHA_MIN:
            self.params.ALPHA *= decay_rate
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

        for ab in self.sub_agents:
            ab.decay(decay_rate)

    def e_greedy_LIA_action(self, state):
        if random.uniform(0, 1) < self.params.EPSILON:
            ab_index = self.random_action()
            action = self.sub_agents[ab_index].random_action()
        else:
            ab_index = self.e_greedy_action(state)
            abs_state = self.encode_abs_state(self.state_decodings[state], self.params.sub_spaces[ab_index])
            action = self.sub_agents[ab_index].greedy_action(abs_state)
        return ab_index, action

    def smart_LiA_choice(self, state):
        if np.random.uniform(0, 1) < self.params.EPSILON:
            return self.random_action()
        max_val = float("-inf")
        max_ab = -1
        state_vars = self.state_decodings[state]
        for ia, ab in enumerate(self.sub_agents):
            abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[ia])
            val = max(ab.Q_table[abs_state])
            if val > max_val:
                max_ab = ia
                max_val = val
        return max_ab  # np.argmax([max(b.Q_table) for b in self.state_bandit_map[state]])

    def update_LIA(self, state, ab_index, action, reward, next_state, done):
        state_vars = self.state_decodings[state]
        next_state_vars = self.state_decodings[next_state]
        for ia, ab in enumerate(self.sub_agents):
            # if ia == ab_index or ia == len(self.sub_agents) - 1:
            abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[ia])
            next_abs_state = self.encode_abs_state(next_state_vars, self.params.sub_spaces[ia])
            ab.update(abs_state, action, reward, next_abs_state, done)
        self.update(state, ab_index, reward, next_state, done)

    def do_step(self):
        state = self.current_state
        ab_index, action = self.e_greedy_LIA_action(state)
        next_state, reward, done = self.step(action)
        # print(self.state_decodings[state], action, reward, next_state, done)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update_LIA(state, ab_index, action, reward, next_state, done)
        self.current_state = next_state
        self.steps += 1
        return reward, done

