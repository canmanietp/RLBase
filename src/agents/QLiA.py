import random

from agents.Q import QAgent
from agents.Q import QMiniAgent
from learning_parameters import DiscreteParameters
import numpy as np
import copy


class QLiAAgent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'LiA'
        self.sub_agents = []
        self.params = params

        for ab in params.sub_spaces:
            ss = 1
            for var in ab:
                ss *= params.size_state_vars[var]

            ab_params = copy.copy(self.params)
            ab_params.EPSILON = params.PHI
            ab_params.EPSILON_MIN = params.PHI_MIN
            self.sub_agents.append(QMiniAgent(self.env, ab_params, ss, self.env.action_space.n))

        self.action_space = len(params.sub_spaces)
        self.Q_table = np.zeros([self.observation_space, self.action_space])

        self.state_decodings = self.sweep_state_decodings()

        self.last_ab = None
        self.last_state = None
        self.next_abstraction = None
        self.next_action = None

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

        for ab in self.sub_agents:
            ab.decay(decay_rate)

    def e_greedy_LIA_action(self, state):
        if random.uniform(0, 1) < self.params.EPSILON:
            ab_index = self.random_action()
            action = self.sub_agents[0].random_action()
        else:
            ab_index = self.greedy_action(state)
            abs_state = self.encode_abs_state(self.state_decodings[state], self.params.sub_spaces[ab_index])
            action = self.sub_agents[ab_index].greedy_action(abs_state)
        return ab_index, action

    def update_LIA(self, state, ab_index, action, reward, next_state, done):
        state_vars = self.state_decodings[state]
        next_state_vars = self.state_decodings[next_state]

        for ia, ab in enumerate(self.sub_agents):
            abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[ia])
            abs_next_state = self.encode_abs_state(next_state_vars, self.params.sub_spaces[ia])
            # lr = self.params.ALPHA / (1 + (1 - int(ia == ab_index))*np.sum(ab.sa_visits[abs_state]))
            # ab.params.ALPHA = lr
            ab.update(abs_state, action, reward, abs_next_state, done)

        self.update(state, ab_index, reward, next_state, done)
        # self.true_agent.update(state, action, reward, next_state, done)

        # self.env.local_reward(state, action, self.params.sub_spaces[ia])

    def do_step(self):
        state = self.current_state
        ab_index, action = self.e_greedy_LIA_action(state)
        next_state, reward, done = self.step(action)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update_LIA(state, ab_index, action, reward, next_state, done)
        self.last_state = state
        self.current_state = next_state
        if done:
            self.last_ab = None
            self.last_state = None
        self.steps += 1
        return reward, done

