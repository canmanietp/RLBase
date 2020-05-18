from agents.base import BaseAgent
from agents.Q import QMiniAgent
from helpers import sensitivity
from collections import deque
import numpy as np
import random, copy


class QSensAgent(BaseAgent):
    def __init__(self, env, params):
        super().__init__(env)
        self.name = 'Q_sens'
        self.params = copy.copy(params)
        self.Q_table = np.zeros([self.observation_space, self.action_space])

        self.state_variables = list(range(len(self.params.size_state_vars)))
        self.sub_agents = []

        for sv in self.state_variables:
            ss = 1
            for var in [value for value in self.state_variables if value != sv]:
                ss *= params.size_state_vars[var]
            self.sub_agents.append(QMiniAgent(self.env, params, ss, self.env.action_space.n))

        self.state_ab_mapping = [[] for os in range(self.observation_space)]

        self.history = [] # deque(maxlen=5)
        self.state_decodings = self.sweep_state_decodings()

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

    def e_greedy_action(self, state):
        if random.uniform(0, 1) < self.params.EPSILON:
            return self.random_action()
        return np.argmax(self.Q_table[state])

    def decay(self, decay_rate):
        if self.params.ALPHA > self.params.ALPHA_MIN:
            self.params.ALPHA *= decay_rate
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def update(self, state, action, reward, next_state, done):
        if done:
            td_error = reward - self.Q_table[state][action]
        else:
            td_error = reward + self.params.DISCOUNT * max(self.Q_table[next_state]) - self.Q_table[state][action]

        self.Q_table[state][action] += self.params.ALPHA * td_error

        # if len(self.state_ab_mapping[state]) > 0:
        state_vars = self.state_decodings[state]
        next_state_vars = self.state_decodings[next_state]
        for ia, ab in enumerate(self.sub_agents):
            ab_vars = [value for value in self.state_variables if value != ia]
            abs_state = self.encode_abs_state(state_vars, ab_vars)
            abs_next_state = self.encode_abs_state(next_state_vars, ab_vars)
        #     # lr = self.params.ALPHA / (1 + (1 - ia == ab_index) * self.sa_visits[state][ia])
            # ab.params.ALPHA = lr
            ab.update(abs_state, action, reward, abs_next_state, done)

        return td_error

    def do_sens_analysis(self):
        sensitivities = sensitivity.do_sensitivity_analysis(self, self.trajectory, self.state_variables)
        traj_sum = [0 for sv in self.state_variables]
        for s, _, _, _ in self.trajectory:
            traj_sum = np.add(traj_sum, sensitivities[s])
        least_influence = np.argmax(traj_sum)
        for s, _, _, _ in self.trajectory:
            self.state_ab_mapping[s] = [least_influence]
        return

    def do_step(self):
        state = self.current_state

        if 5 < np.sum(self.sa_visits[state]) < 20:  # random.uniform(0, 1) < self.params.EPSILON:
            # if self.state_ab_mapping[state] == []:
                # action = np.argmax(self.Q_table[state])
                # next_state, reward, done = self.step(action)
            # else:
            sensitivities = sensitivity.do_sensitivity_analysis_single_state(self, self.history, state, self.state_variables)
            least_influence = np.argmax(sensitivities) # self.state_ab_mapping[state][0]  #  np.argmax(traj_sum)
            state_vars = self.state_decodings[state]
            ab_vars = [value for value in self.state_variables if value != least_influence]
            abs_state = self.encode_abs_state(state_vars, ab_vars)
            action = np.argmax(self.sub_agents[least_influence].Q_table[abs_state])
            next_state, reward, done = self.step(action)
        else:
            action = self.e_greedy_action(state) # np.argmax(self.Q_table[state])

        next_state, reward, done = self.step(action)
        self.update(state, action, reward, next_state, done)
        self.history.append([state, action, reward, next_state])
        return reward, done
