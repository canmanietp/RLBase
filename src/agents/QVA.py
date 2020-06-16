import operator

from agents.Q import QAgent
from agents.Q import QMiniAgent
from learning_parameters import DiscreteParameters
import numpy as np
import random
import copy


class QVAAgent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'QVA'
        self.action_space = self.env.action_space.n
        meta_params = self.params
        # meta_params.EPSILON = meta_params.PHI
        # meta_params.EPSILON_MIN = meta_params.PHI_MIN
        self.meta_agent = QMiniAgent(self.env, meta_params, self.observation_space, len(params.sub_spaces))
        self.Q_table = np.zeros([self.observation_space, self.action_space])

        self.saved_abs_lookup = {}
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
        if self.params.ALPHA > self.params.ALPHA_MIN:
            self.params.ALPHA *= decay_rate
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate
        if self.params.PHI > self.params.PHI_MIN:
            self.params.PHI *= decay_rate

        self.meta_agent.decay(decay_rate)

    def legacy_action(self, state):
        ab_index = self.meta_agent.e_greedy_action(state)

        if random.uniform(0, 1) < self.params.PHI:
            return ab_index, self.random_action()
        else:
            abstraction = self.params.sub_spaces[ab_index]
            update_states = []
            merge_values = []
            merge_visits = []

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

            if len(update_states) == 1:
                qs = merge_values[0]
            else:
                # take best action by summing the ranks?
                # BAD
                # ranks_sum = np.zeros(len(merge_values[0]))
                # for im, m in enumerate(merge_values):
                #     temp = m.argsort()
                #     ranks = np.empty_like(temp)
                #     ranks[temp] = np.arange(len(m))
                #     ranks_sum = ranks_sum + ranks
                #
                # return ab_index, np.argmin(ranks_sum)
                # -----
                # VERY BAD
                # best_acts = []
                # for im, m in enumerate(merge_values):
                #     temp = m.argsort()
                #     ranks = np.empty_like(temp)
                #     ranks[temp] = np.arange(len(m))
                #     best_acts.append(np.argmin(ranks))
                #
                # return ab_index, max(set(best_acts), key = best_acts.count)
                # ---------
                # THIS IS OK
                # best_acts = []
                # for im, m in enumerate(merge_values):
                #     best_act = np.argmax(m)
                #     best_acts.append(best_act)
                #
                # if best_acts == []:
                #     return ab_index, self.random_action()
                #
                # return ab_index, max(set(best_acts), key=best_acts.count)
                # ----------
                # qs = np.zeros(self.action_space)
                # for a in range(self.action_space):
                #     b = np.array([item[a] for item in merge_visits])
                #     most_visited = int(np.random.choice(np.flatnonzero(b == b.max())))
                #     qs[a] = merge_values[most_visited][a]
                # ---------
                most_visited = -1
                most_visits = float("-inf")
                for im, m in enumerate(merge_visits):  # try visits to best action
                    if np.sum(m) > most_visits:  # m[np.argmax(merge_values[im])] OR np.sum(m)
                        most_visited = im
                        most_visits = np.sum(m)
                if most_visits == 0:
                    qs = merge_values[most_visited]
                else:
                    return ab_index, self.random_action()
                # ---------
                # most_valued = 0
                # highest_value = 0
                # for im, m in enumerate(merge_values):
                #     if np.max(m) > highest_value:  # np.sum(m)
                #         most_valued = np.argmax(m)  # im
                #         highest_value = np.max(m)
                # return ab_index, most_valued
            return ab_index, np.argmax(qs)

    def update_VA(self, state, ab_index, action, reward, next_state, done):
        # p = 400
        # self.meta_agent.params.ALPHA = self.params.ALPHA / (1 + np.sum(self.sa_visits[state]) / p)
        self.meta_agent.sarsa_update(state, ab_index, reward, next_state, done)
        self.sarsa_update(state, action, reward, next_state, done)

    def do_step(self):
        state = self.current_state
        ab_index, action = self.legacy_action(state)
        next_state, reward, done = self.step(action)
        self.update_VA(state, ab_index, action, reward, next_state, done)
        self.current_state = next_state
        if self.steps > self.max_steps:
            done = True
        return reward, done
