from agents.Q import QAgent
from agents.Q import QMiniAgent
from learning_parameters import DiscreteParameters
import numpy as np
import copy


class QLiA_batchAgent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'LiA batch'
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

        self.batch = []
        self.steps_to_interval = 0
        self.batch_interval = 1000
        self.next_abstraction = None

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


    def LIA_action(self, state):
        if self.next_abstraction:
            ab_index = self.next_abstraction
        else:
            ab_index = self.greedy_action(state)

        abs_state = self.encode_abs_state(self.state_decodings[state], self.params.sub_spaces[ab_index])
        action = self.sub_agents[ab_index].e_greedy_action(abs_state)
        return ab_index, action

    def remember(self, state, action, reward, next_state, done):
        self.batch.append([state, action, reward, next_state, done])

    def batch_update(self):
        for s, a, r, ns, done in self.batch:
            self.update(s, a, r, ns, done)
            if done:
                self.decay(0.9)

    def update_LIA(self, state, ab_index, action, reward, next_state, done):
        state_vars = self.state_decodings[state]
        next_state_vars = self.state_decodings[next_state]

        print(state_vars, ab_index, action)

        self.next_abstraction = self.e_greedy_action(next_state)
        next_abs_state = self.encode_abs_state(next_state_vars, self.params.sub_spaces[self.next_abstraction])
        for ia, ab in enumerate(self.sub_agents):
            abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[ia])
            td_error = reward + (not done) * self.params.DISCOUNT * max(self.sub_agents[self.next_abstraction].Q_table[next_abs_state]) - ab.Q_table[abs_state][action]
            ab.Q_table[abs_state][action] += self.params.ALPHA * td_error
            if done:
                ab.decay(self.params.DECAY_RATE)
        self.remember(state, ab_index, reward, next_state, done)

        if self.steps_to_interval > self.batch_interval:
            print("batch updating ...")
            self.batch_update()
            self.batch = []
            self.steps_to_interval = 0
            self.sub_agents = []
            for ab in self.params.sub_spaces:
                ss = 1
                for var in ab:
                    ss *= self.params.size_state_vars[var]

                ab_params = copy.copy(self.params)
                ab_params.EPSILON = 0.3
                ab_params.EPSILON_MIN = self.params.EPSILON_MIN
                self.sub_agents.append(QMiniAgent(self.env, ab_params, ss, self.env.action_space.n))
            print("continuing.")
            if self.params.EPSILON > self.params.EPSILON_MIN:
                self.params.EPSILON *= self.params.DECAY_RATE

    def do_step(self):
        state = self.current_state
        ab_index, action = self.LIA_action(state)
        next_state, reward, done = self.step(action)
        self.update_LIA(state, ab_index, action, reward, next_state, done)
        if done:
            self.steps_to_interval += 1
        return reward, done

