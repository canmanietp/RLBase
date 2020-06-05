from agents.Q import QAgent
from agents.Q import QMiniAgent
from learning_parameters import DiscreteParameters
import numpy as np
import random
import copy
from helpers import sensitivity
from helpers import ranges
from collections import deque


class QVPAgent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'QVP'
        self.action_space = self.env.action_space.n
        meta_params = self.params
        # meta_params.EPSILON = meta_params.PHI
        # meta_params.EPSILON_MIN = meta_params.PHI_MIN
        self.meta_agent = QMiniAgent(self.env, meta_params, self.observation_space, len(params.sub_spaces))
        self.Q_table = np.zeros([self.observation_space, self.action_space])

        self.saved_abs_lookup = {}
        self.state_decodings = self.sweep_state_decodings()

        self.state_variables = list(range(len(self.params.size_state_vars)))
        self.ranges = ranges.get_var_ranges(self, [np.zeros(self.observation_space), self.params.size_state_vars],
                                            self.state_variables)
        self.memory_length = 3
        self.trajectory = deque(maxlen=self.memory_length)

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
        if self.params.PHI > self.params.PHI_MIN:
            self.params.PHI *= decay_rate

        self.meta_agent.decay(decay_rate)

    def e_greedy_ignore_action(self, state):
        if np.random.uniform(0, 1) > self.params.EPSILON or all([x == 0. for x in self.Q_table[state]]):
            if np.sum(self.sa_visits[state]) == 0:
                most_visits = float("-inf")
                qs = []
                ignoring = -1
                for v in self.state_variables:
                    abstraction = [value for value in self.state_variables if value != v]

                    update_states = []
                    merge_values = []
                    merge_visits = []

                    if state not in self.saved_abs_lookup:
                        self.saved_abs_lookup[state] = {}

                    if v not in self.saved_abs_lookup[state]:
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
                                merge_values.append(self.Q_table[st])
                                merge_visits.append(self.sa_visits[st])

                        if not merge_visits:
                            update_states.append(state)
                            merge_values.append(self.Q_table[state])
                            merge_visits.append(self.sa_visits[state])

                        self.saved_abs_lookup[state][v] = update_states
                    else:
                        for us in self.saved_abs_lookup[state][v]:
                            update_states.append(us)
                            merge_values.append(self.Q_table[us])
                            merge_visits.append(self.sa_visits[us])

                    if len(update_states) == 1:
                        qs = merge_values[0]
                        # qs = self.Q_table[update_states[0]]
                    else:
                        merge_values = []
                        merge_visits = []

                        for st in update_states:
                            merge_values.append(self.Q_table[st])
                            merge_visits.append(self.sa_visits[st])

                        visited = 0
                        visits = 0
                        for im, m in enumerate(merge_visits):
                            num_visits = np.sum(m)
                            if num_visits > visits:
                                visited = im
                                visits = num_visits

                        if visits > most_visits:
                            most_visits = visits
                            qs = merge_values[visited]
                            ignoring = v

                        # if np.sum(merge_visits) > most_visits:
                        #     most_visits = np.sum(merge_visits)
                        #     qs = np.sum(merge_values, axis=0)
                        #     ignoring = v
                #
                # if np.argmax(qs) != np.argmax(self.Q_table[state]):
                #     print("in state ", list(self.env.decode(state)), "and ignoring", ignoring, "taking action", np.argmax(qs), "instead of", self.greedy_action(state))
                return True, np.argmax(qs)
            else:
                return False, self.greedy_action(state)
        else:
            return False, self.random_action()

    def e_greedy_VP_action(self, state):
        if np.random.uniform(0, 1) > self.params.EPSILON or all([x == 0 for x in self.Q_table[state]]):
            if len(self.trajectory) >= self.memory_length and 500 > np.sum(self.sa_visits[state]) > 50:
                # least_influence, update_states = sensitivity.do_sensitivity_analysis(self, self.ranges, self.trajectory, self.state_variables)
                least_influence, update_states = sensitivity.do_sensitivity_analysis_single_state(self, self.ranges,
                                                                                                  state,
                                                                                                  self.state_variables)

                if least_influence is not None:
                    abstraction = [value for value in self.state_variables if value != least_influence]

                    update_states = []
                    merge_values = []
                    merge_visits = []

                    if state not in self.saved_abs_lookup:
                        self.saved_abs_lookup[state] = {}

                    if least_influence not in self.saved_abs_lookup[state]:
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
                                merge_values.append(self.Q_table[st])
                                merge_visits.append(self.sa_visits[st])

                        if not merge_visits:
                            update_states.append(state)
                            merge_values.append(self.Q_table[state])
                            merge_visits.append(self.sa_visits[state])

                        self.saved_abs_lookup[state][least_influence] = update_states
                    else:
                        for us in self.saved_abs_lookup[state][least_influence]:
                            update_states.append(us)
                            merge_values.append(self.Q_table[us])
                            merge_visits.append(self.sa_visits[us])

                    if len(update_states) == 1:
                        qs = merge_values[0]
                        # qs = self.Q_table[update_states[0]]
                    else:
                        merge_values = []
                        merge_visits = []

                        for st in update_states:
                            merge_values.append(self.Q_table[st])
                            merge_visits.append(self.sa_visits[st])

                        most_visited = 0
                        most_visits = 0
                        for im, m in enumerate(merge_visits):
                            num_visits = np.sum(m)
                            if num_visits > most_visits:
                                most_visited = im
                                most_visits = num_visits
                        qs = merge_values[most_visited]
                        # qs = []
                        # max_val = float("-inf")
                        # for v in merge_values:
                        #     if np.max(v) >= max_val:
                        #         qs = v
                        #         max_val = np.max(v)

                    # if np.argmax(qs) != np.argmax(self.Q_table[state]):
                    #     print("in state ", list(self.env.decode(state)), "and ignoring", least_influence, "taking action", np.argmax(qs), "instead of", np.argmax(self.Q_table[state]))
                    return True, np.argmax(qs)
                else:
                    return False, np.argmax(self.Q_table[state])
            else:
                return False, np.argmax(self.Q_table[state])
        else:
            return False, self.random_action()  # np.argmax(self.Q_table[state])

    def update_VP(self, state, action, reward, next_state, done):
        self.update(state, action, reward, next_state, done)

    def do_step(self):
        state = self.current_state
        used_QVP, action = self.e_greedy_ignore_action(state)
        next_state, reward, done = self.step(action)
        self.trajectory.append([state, action, reward, next_state])
        self.update_VP(state, action, reward, next_state, done)
        self.current_state = next_state
        if done:
            self.trajectory.clear()

        self.steps += 1
        return reward, done
