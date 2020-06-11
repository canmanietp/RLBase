from agents.Q import QAgent
import numpy as np
import random


class QBiasAgent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = "QBias"
        self.biases = np.zeros([self.observation_space, self.action_space])
        self.saved_abs_lookup = {}
        self.state_decodings = self.sweep_state_decodings()

    def sweep_state_decodings(self):
        st_vars_lookup = []
        for s in range(self.observation_space):
            st_vars_lookup.append(list(self.env.decode(s)))
        return st_vars_lookup

    def e_greedy_action(self, state):
        if random.uniform(0, 1) < self.params.EPSILON:
            return self.random_action()

        if np.sum(self.biases[state]) > 0:  # i.e. there is a biased action
            return np.argmax(self.biases[state])

        if all([x == 0 for x in self.Q_table[state]]):
            return self.random_action()
        return self.greedy_action(state)

    def bias_actions(self, state, action):
        state_vars = list(self.env.decode(state))
        abstraction = [0, 1, 2] if state_vars[2] < 4 else [0, 1, 2, 3]
        # abstraction = [0, 1, 3, 6, 7] if state_vars[4] == 0 and state_vars[5] == 0 else [0, 1, 2, 4, 5]
        ab_index = 0 if state_vars[2] < 4 else 1
        # ab_index = 0 if state_vars[4] == 0 and state_vars[5] == 0 else 1
        update_states = []
        bias = np.zeros(self.action_space)
        bias[action] = self.sa_visits[state][action]

        if state not in self.saved_abs_lookup:
            self.saved_abs_lookup[state] = {}

        if ab_index not in self.saved_abs_lookup[state]:
            state_vars = self.state_decodings[state]
            for st in range(self.observation_space):
                st_vars = self.state_decodings[st]
                is_valid = True

                for av in abstraction:
                    if not state_vars[av] == st_vars[av]:
                        is_valid = False
                        break

                if is_valid:
                    update_states.append(st)
                    self.biases[st] = bias
            self.saved_abs_lookup[state][ab_index] = update_states
        else:
            for us in self.saved_abs_lookup[state][ab_index]:
                self.biases[us] = bias

    def do_step(self):
        state = self.current_state
        action = self.e_greedy_action(state)
        next_state, reward, done = self.step(action)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update(state, action, reward, next_state, done)
        self.bias_actions(state, action)
        self.current_state = next_state
        self.steps += 1
        return reward, done

