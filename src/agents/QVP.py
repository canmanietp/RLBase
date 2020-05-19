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
        meta_params.EPSILON = meta_params.PHI
        meta_params.EPSILON_MIN = meta_params.PHI_MIN
        self.meta_agent = QMiniAgent(self.env, meta_params, self.observation_space, len(params.sub_spaces))
        self.Q_table = np.zeros([self.observation_space, self.action_space])
        self.saved_abs_lookup = {}

        self.state_decodings = self.sweep_state_decodings()
        self.num_visits = 0

        self.state_variables = list(range(len(self.params.size_state_vars)))
        self.ranges = ranges.get_var_ranges(self, [np.zeros(self.observation_space), self.params.size_state_vars], self.state_variables)
        self.trajectory = deque(maxlen=10)

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

    def e_greedy_VP_action(self, state):
        # ab_index = self.meta_agent.e_greedy_action(state)
        # if random.uniform(0, 1) < self.params.PHI:
        #     return ab_index, self.random_action()
        # else:
        # abstraction = self.params.sub_spaces[ab_index]
        # update_states = []
        # merge_values = []
        # merge_visits = []
        #
        # if state not in self.saved_abs_lookup:
        #     self.saved_abs_lookup[state] = {}
        #
        # # if ab_index not in self.saved_abs_lookup[state]:
        # state_vars = self.state_decodings[state]
        # for st in range(self.observation_space):
        #     st_vars = self.state_decodings[st]
        #     is_valid = True
        #
        #     for av in abstraction:
        #         if not state_vars[av] == st_vars[av]:
        #             is_valid = False
        #             break
        #
        #     # if is_valid:
        #     #     # rank actions
        #     #     array1 = self.Q_table[state]
        #     #     temp = array1.argsort()
        #     #     ranks1 = np.empty_like(temp)
        #     #     ranks1[temp] = np.arange(len(array1))
        #     #     array2 = self.Q_table[st]
        #     #     temp = array1.argsort()
        #     #     ranks2 = np.empty_like(temp)
        #     #     ranks2[temp] = np.arange(len(array2))
        #     #
        #     #     if not (ranks1 == ranks2).all():
        #     #         is_valid = False
        #
        #     if is_valid:
        #         update_states.append(st)
        #         merge_values.append(self.Q_table[st])
        #         merge_visits.append(self.sa_visits[st])
        #
        # if not merge_visits:
        #     update_states.append(state)
        #     merge_values.append(self.Q_table[state])
        #     merge_visits.append(self.sa_visits[state])
        #
        # # self.saved_abs_lookup[state][ab_index] = update_states
        # # else:
        # #     for us in self.saved_abs_lookup[state][ab_index]:
        # #         update_states.append(us)
        # #         merge_values.append(self.Q_table[us])
        # #         merge_visits.append(self.sa_visits[us])
        #
        # if len(update_states) == 1:
        #     qs = merge_values[0]
        # else:
        #     qs = np.zeros(self.action_space)
        #     for a in range(self.action_space):
        #         b = np.array([item[a] for item in merge_visits])
        #         most_visited = int(np.random.choice(np.flatnonzero(b == b.max())))
        #         qs[a] = merge_values[most_visited][a]
        # return ab_index, np.argmax(qs)

        if np.sum(self.sa_visits[state]) < 10:
            sensitivities = sensitivity.do_sensitivity_analysis(self, self.ranges, self.trajectory, self.state_variables)
            traj_sum = [0 for sv in self.state_variables]
            for s, _, _, _ in self.trajectory:
                traj_sum = np.add(traj_sum, sensitivities[s])
            least_influence = np.argmax(traj_sum)
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
            else:
                qs = np.zeros(self.action_space)
                for a in range(self.action_space):
                    b = np.array([item[a] for item in merge_visits])
                    most_visited = int(np.random.choice(np.flatnonzero(b == b.max())))
                    qs[a] = merge_values[most_visited][a]

            return None, np.argmax(qs)
        else:
            return None, self.e_greedy_action(state)

        # if np.random.uniform(0, 1) < self.params.EPSILON:
        #     sensitivities = sensitivity.do_sensitivity_analysis_single_state(self, self.ranges, state,
        #                                                         self.state_variables)
        #     least_influence = int(np.argmax(sensitivities))
        #     if sensitivities[least_influence] > 0.: # 2.0 * np.std(sensitivities) + np.mean(sensitivities):
        #         abstraction = [value for value in self.state_variables if value != least_influence]
        #
        #         update_states = []
        #         merge_values = []
        #         merge_visits = []
        #
        #         state_vars = self.state_decodings[state]
        #         for st in range(self.observation_space):
        #             st_vars = self.state_decodings[st]
        #             is_valid = True
        #
        #             for av in abstraction:
        #                 if not state_vars[av] == st_vars[av]:
        #                     is_valid = False
        #                     break
        #
        #             if is_valid:
        #                 update_states.append(st)
        #                 merge_values.append(self.Q_table[st])
        #                 merge_visits.append(self.sa_visits[st])
        #
        #         if not merge_visits:
        #             update_states.append(state)
        #             merge_values.append(self.Q_table[state])
        #             merge_visits.append(self.sa_visits[state])
        #
        #         if len(update_states) == 1:
        #             qs = merge_values[0]
        #         else:
        #             qs = np.zeros(self.action_space)
        #             for a in range(self.action_space):
        #                 b = np.array([item[a] for item in merge_visits])
        #                 most_visited = int(np.random.choice(np.flatnonzero(b == b.max())))
        #                 qs[a] = merge_values[most_visited][a]
        #         return None, np.argmax(qs)
        #     else:
        #         return None, self.random_action()
        # else:
        #     return None, np.argmax(self.Q_table[state])

    def update_VP(self, state, ab_index, action, reward, next_state, done):
        self.update(state, action, reward, next_state, done)
        if ab_index is not None:
            self.meta_agent.update(state, ab_index, reward, next_state, done)

    def do_step(self):
        state = self.current_state
        # confident = self.sa_visits[state][np.argmax(self.Q_table[state])] > np.percentile(self.num_visits, 80)
        # if confident:
        #     action = self.e_greedy_action(state)
        #     next_state, reward, done = self.step(action)
        #     self.update(state, action, reward, next_state, done)
        # else:
        ab_index, action = self.e_greedy_VP_action(state)
        next_state, reward, done = self.step(action)
        self.trajectory.append([state, action, reward, next_state])
        self.update_VP(state, ab_index, action, reward, next_state, done)
        # if done:
        #     self.num_visits = np.array(self.sa_visits)[:, 1]
        return reward, done
