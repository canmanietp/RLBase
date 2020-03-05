from Agents.Q import QAgent
from Agents.Q import QMiniAgent
import numpy as np
import random
import copy


class QLiAAgent(QAgent):
    def __init__(self, env, alpha, alpha_min, epsilon, epsilon_min, discount, size_state_vars, abstractions, phi, phi_min):
        super().__init__(env, alpha, alpha_min, epsilon, epsilon_min, discount)
        self.name = 'LiA'
        self.PHI = phi
        self.PHI_MIN = phi_min
        self.abstractions = abstractions
        self.size_state_vars = size_state_vars
        self.abstraction_agents = []

        for ab in abstractions:
            ss = 1
            for var in ab:
                ss *= self.size_state_vars[var]
            self.abstraction_agents.append(QMiniAgent(self.env, self.ALPHA, self.ALPHA_MIN, self.PHI, self.PHI_MIN, self.DISCOUNT, ss, self.env.action_space.n))

        self.action_space = len(abstractions)
        self.Q_table = np.zeros([self.observation_space, self.action_space])

        self.state_decodings = self.sweep_state_decodings()

    def sweep_state_decodings(self):
        st_vars_lookup = []
        for s in range(self.observation_space):
            st_vars_lookup.append(list(self.env.decode(s)))
        return st_vars_lookup

    def encode_abs_state(self, state, abstraction):
        abs_state = [state[k] for k in abstraction]
        var_size = copy.copy([self.size_state_vars[k] for k in abstraction])
        var_size.pop(0)
        encoded_state = 0

        for e in range(len(abs_state) - 1):
            encoded_state += abs_state[e] * np.prod(var_size)
            var_size.pop(0)

        encoded_state += abs_state[-1]
        return encoded_state

    def decay(self, decay_rate):
        if self.ALPHA > self.ALPHA_MIN:
            self.ALPHA *= decay_rate
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= decay_rate

        for ab in self.abstraction_agents:
            ab.decay(decay_rate)

    def e_greedy_LIA_action(self, state):
        ab_index = self.e_greedy_action(state)
        abs_state = self.encode_abs_state(self.state_decodings[state], self.abstractions[ab_index])
        action = self.abstraction_agents[ab_index].e_greedy_action(abs_state)
        return ab_index, action

    def update_LIA(self, state, ab_index, action, reward, next_state, done):
        state_vars = self.state_decodings[state]
        next_state_vars = self.state_decodings[next_state]

        for ia, ab in enumerate(self.abstraction_agents):
            abs_state = self.encode_abs_state(state_vars, self.abstractions[ia])
            abs_next_state = self.encode_abs_state(next_state_vars, self.abstractions[ia])
            ab.update(abs_state, action, reward, abs_next_state, done)

        self.update(state, ab_index, reward, next_state, done)

    def run_episode(self):
        state = self.current_state
        ab_index, action = self.e_greedy_LIA_action(state)
        next_state, reward, done = self.step(action)
        self.update_LIA(state, ab_index, action, reward, next_state, done)
        return reward, done

