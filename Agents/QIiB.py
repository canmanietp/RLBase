from Agents.Q import QAgent
from Agents.Q import QMiniAgent
import numpy as np
import random
import copy


class QIiBAgent(QAgent):
    def __init__(self, env, alpha, alpha_min, epsilon, epsilon_min, discount, size_state_vars, abstractions, phi, phi_min):
        super().__init__(env, alpha, alpha_min, epsilon, epsilon_min, discount)
        self.name = 'IiB'
        self.PHI = phi
        self.PHI_MIN = phi_min
        self.size_state_vars = size_state_vars
        self.abstractions = abstractions
        self.action_space = self.env.action_space.n
        self.meta_agent = QMiniAgent(self.env, self.ALPHA, self.ALPHA_MIN, self.PHI, self.PHI_MIN, self.DISCOUNT, self.observation_space, len(abstractions))
        self.Q_table = np.zeros([self.observation_space, self.action_space])
        self.saved_abs_lookup = {}

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
        if self.PHI > self.PHI_MIN:
            self.PHI *= decay_rate

    def e_greedy_IiB_action(self, state):
        ab_index = self.meta_agent.e_greedy_action(state)

        if random.uniform(0, 1) < self.PHI:
            return ab_index, self.random_action()
        else:
            update_states = []
            merge_values = []
            merge_visits = []
            abstraction = self.abstractions[ab_index]

            if state not in self.saved_abs_lookup:
                self.saved_abs_lookup[state] = {}

            if ab_index not in self.saved_abs_lookup[state]:
                state_vars = list(self.env.decode(state))
                for st in range(self.observation_space):
                    st_vars = list(self.env.decode(st))
                    is_valid = True

                    for av in abstraction:
                        if not state_vars[av] == st_vars[av]:
                            is_valid = False
                            break

                    if is_valid:
                        update_states.append(st)
                        merge_values.append(self.Q_table[st])
                        merge_visits.append(self.sa_visits[st])

                if not merge_visits:
                    update_states.append(state)
                    merge_values.append(self.Q_table[state])
                    merge_visits.append(self.sa_visits[state])

                self.saved_abs_lookup[state][ab_index] = update_states
            else:
                for us in self.saved_abs_lookup[state][ab_index]:
                    update_states.append(us)
                    merge_values.append(self.Q_table[us])
                    merge_visits.append(self.sa_visits[us])

            qs = np.zeros(self.action_space)

            for a in range(self.action_space):
                most_visited = int(np.argmax([item[a] for item in merge_visits]))
                qs[a] = merge_values[most_visited][a]

            return ab_index, np.argmax(qs)

    def update_IiB(self, state, ab_index, action, reward, next_state, done):
        self.update(state, action, reward, next_state, done)
        self.meta_agent.update(state, ab_index, reward, next_state, done)

    def run_episode(self):
        state = self.current_state
        ab_index, action = self.e_greedy_IiB_action(state)
        next_state, reward, done = self.step(action)
        self.update_IiB(state, ab_index, action, reward, next_state, done)
        return reward, done
