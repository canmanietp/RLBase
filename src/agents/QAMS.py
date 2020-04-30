# Q-learning with abstraction model selection
from agents.Q import QAgent
from agents.Q import QMiniAgent
from learning_parameters import DiscreteParameters
import numpy as np
import copy


class QAMSAgent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'QAMS'
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

        self.sub_agent_error = [np.zeros([len(self.sub_agents), self.action_space]) for os in range(self.observation_space)]
        # self.action_space = len(self.sub_agents)
        self.Q_table = np.zeros([self.observation_space, self.action_space])

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

    def decay(self, decay_rate):
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate
        if self.params.ALPHA > self.params.ALPHA_MIN:
            self.params.ALPHA *= decay_rate

        for ab in self.sub_agents:
            ab.decay(decay_rate)

    def e_greedy_AMS_action(self, state):
        sums = []
        for i, state_errors in enumerate(self.sub_agent_error[state]):
            for sub_agent in state_errors:
                if i == 0:
                    sums.append(np.sum(sub_agent))
                else:
                    sums[i] = np.sum(sub_agent)

        print(sums, int(np.argmax(sums)))

        ab_index = int(np.argmax(sums))
        abs_state = self.encode_abs_state(self.state_decodings[state], self.params.sub_spaces[ab_index])
        return ab_index, self.sub_agents[ab_index].e_greedy_action(abs_state)

    def update_AMS(self, state, ab_index, action, reward, next_state, done):
        state_vars = self.state_decodings[state]
        next_state_vars = self.state_decodings[next_state]

        ia = ab_index
        ab = self.sub_agents[ia]
        # for ia, ab in enumerate(self.sub_agents):
        abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[ia])
        abs_next_state = self.encode_abs_state(next_state_vars, self.params.sub_spaces[ia])

        if done:
            td_error = reward - ab.Q_table[abs_state][action]
        else:
            td_error = reward + ab.params.DISCOUNT * max(ab.Q_table[abs_next_state]) - ab.Q_table[abs_state][action]

        ab.Q_table[abs_state][action] += ab.params.ALPHA * td_error
        self.sub_agent_error[state][ia][action] = ab.Q_table[abs_state][action]  # reward # / td_error

        self.update(state, action, reward, next_state, done)

    def do_step(self):
        state = self.current_state
        ab_index, action = self.e_greedy_AMS_action(state)
        next_state, reward, done = self.step(action)
        self.update_AMS(state, ab_index, action, reward, next_state, done)
        return reward, done

