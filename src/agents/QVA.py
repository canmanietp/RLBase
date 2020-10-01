from agents.Q import QAgent
from agents.Q import QMiniAgent
from agents.LOARA_known import LOARA_K_Agent
import numpy as np
import copy
import pickle


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

        # infile = open('helpers/state_mapping_{}'.format(str(env)), 'rb')
        # self.state_mapping = pickle.load(infile)

        self.values_for = {}

        self.values_for_221 = []
        self.values_for_202 = []
        self.values_for_313 = []
        self.values_for_422 = []

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

    def return_abstract_state(self, state):
        state_vars = self.state_decodings[state]
        sv = list(state_vars)
        for ir in self.state_mapping:
            if sv in ir:
                return ir
        return None

    def heuristic_abstraction_choice(self, state):
        state_vars = self.state_decodings[state]
        if 'TaxiFuel' in str(self.env):
            if state_vars[2] < 4 and state_vars[4] > 12:  # don't have passenger and have enough fuel to get any
                # passenger to any destination
                return self.params.sub_spaces.index([0, 1, 2])
            elif state_vars[2] == 4 and state_vars[4] > 8:  # have passenger and have enough fuel to get to any
                # destination
                return self.params.sub_spaces.index([0, 1, 2, 3])
            elif state_vars[2] == 4 and (state_vars[3] == 0 or state_vars[3] == 2) and state_vars[1] == 0 and \
                    state_vars[4] > 4:
                return self.params.sub_spaces.index([0, 1, 2, 3])
            elif state_vars[2] == 4 and (state_vars[3] == 1 or state_vars[3] == 3) and state_vars[1] >= 3 and \
                    state_vars[4] > 4:
                return self.params.sub_spaces.index([0, 1, 2, 3])
            else:
                return len(self.params.sub_spaces) - 1
        elif 'TaxiLarge' in str(self.env):
            if state_vars[1] == 9 and state_vars[2] == 8 and state_vars[3] == 7:
                return self.params.sub_spaces.index([1, 2, 3])
            elif state_vars[1] == 8 and state_vars[2] == 8 and state_vars[3] == 6:
                return self.params.sub_spaces.index([1, 2, 3])
            elif state_vars[1] == 0 and state_vars[2] == 8 and (state_vars[3] == 0 or state_vars[3] == 2):
                return self.params.sub_spaces.index([1, 2, 3])
            elif state_vars[1] == 4 and state_vars[2] == 8 and (state_vars[3] == 1 or state_vars[3] == 3):
                return self.params.sub_spaces.index([1, 2, 3])
            elif state_vars[2] < 8:
                return self.params.sub_spaces.index([0, 1, 2])
            else:
                return len(self.params.sub_spaces) - 1
        elif 'Taxi' in str(self.env):
            if state_vars[0] == 4 and state_vars[1] == 1 or state_vars[0] == 4 and state_vars[1] == 2:
                return self.params.sub_spaces.index([2])
            if state_vars[0] == 3 and state_vars[2] == 4 and state_vars[3] == 0:
                return self.params.sub_spaces.index([0, 2, 3])
            elif state_vars[1] == 4 and state_vars[2] == 4 and state_vars[3] != 1:
                return self.params.sub_spaces.index([1, 2, 3])
            elif state_vars[2] < 4:
                return self.params.sub_spaces.index([0, 1, 2])
            else:
                return len(self.params.sub_spaces) - 1
        elif 'CoffeeMail' in str(self.env):
            # [0, 1, 2, 4, 6], [0, 1, 3, 5, 7],
            if state_vars[4] == 1 or state_vars[5] == 1:
                return self.params.sub_spaces.index([0, 1, 2, 3, 4, 5])
            elif state_vars[6] == 0 or state_vars[7] == 0:
                return self.params.sub_spaces.index([0, 1, 2, 3, 6, 7])
            return len(self.params.sub_spaces) - 1
        elif 'Coffee' in str(self.env):
            if state_vars[2] == 0:
                return self.params.sub_spaces.index([0, 1, 2])
            else:
                return len(self.params.sub_spaces) - 1
        elif 'FourState' in str(self.env):
            if state_vars[0] == 0:
                return self.params.sub_spaces.index([0])
            else:
                return len(self.params.sub_spaces) - 1
        elif 'Warehouse' in str(self.env):
            if state_vars[0] < len(self.env.locs) - 1 and state_vars[state_vars[0] + 1] > 0:
                # print(state_vars, [0, state_vars[0] + 1])
                return self.params.sub_spaces.index([0, state_vars[0] + 1])
            if np.sum(state_vars[1:]) <= 1:
                return self.params.sub_spaces.index([*range(1, self.env.num_products + 1)])
            return len(self.params.sub_spaces) - 1
        else:
            print("ERROR: UNKNOWN HEURISTIC FOR CHOOSING ABSTRACTION")
            quit()

    def percent_diff_action(self, state):
        if np.random.uniform(0, 1) < self.params.EPSILON:
            return 0, self.random_action()

        if np.random.uniform(0, 1) < min([1, 5. / (1 + np.sum(self.sa_visits[state]))]):
            abstract_states = self.return_abstract_state(state)

            if abstract_states is None:
                return 0, self.greedy_action(state)

            values = []
            visits = []
            encoded_states = []
            for us in abstract_states:
                es = self.env.encode(*tuple(int(u) for u in us))
                if np.sum(self.sa_visits[es]) > 0:
                    values.append(max(self.Q_table[es]))
                    visits.append(np.sum(self.sa_visits[es]))
                    encoded_states.append(es)

            return None, np.argmax(
                self.Q_table[encoded_states[int(np.argmax(values))]]) if encoded_states else self.greedy_action(state)
        else:
            return None, self.greedy_action(state)

        # abstract_states = self.return_abstract_state(state)
        #
        # if abstract_states is None:
        #     return 0, self.greedy_action(state)
        # yes = 0
        # total = 0
        # action = np.argmax(self.Q_table[state])
        #
        # for us in abstract_states:
        #     encoded_us = self.env.encode(*tuple(int(u) for u in us))
        #     if self.sa_visits[encoded_us][action] > 0:
        #         if np.argmax(self.Q_table[encoded_us]) == action:
        #             yes += 1
        #         total += 1
        #
        # percent_agreeance = yes / total if total > 0 else 0
        # # print(self.state_decodings[state], action, percent_agreeance)
        # return percent_agreeance, action

        # qs = copy.copy(self.Q_table[state])
        # weighted_one_hot = np.zeros(self.action_space)
        # beta = 3. / (1 + np.sum(self.sa_visits[state]))
        #
        # for im, mv in enumerate(merge_values):
        #     q = np.zeros(self.action_space)
        #     q[np.argmax(mv)] = 1 * self.sa_visits[update_states[im]][np.argmax(mv)]
        #     weighted_one_hot += q
        #
        # # qs += beta * weighted_one_hot
        # if np.random.uniform(0, 1) < self.params.EPSILON:
        #     qs = weighted_one_hot
        #
        # return None, np.argmax(qs), None

    def legacy_action(self, state):
        if np.random.uniform(0, 1) < self.params.EPSILON:
            return self.random_action()

        ab_index = self.heuristic_abstraction_choice(state)
        abstraction = self.params.sub_spaces[ab_index]
        update_states = []
        merge_values = []
        merge_visits = []

        if state not in self.saved_abs_lookup:
            self.saved_abs_lookup[state] = {}

        if len(self.params.sub_spaces[ab_index]) < len(self.params.size_state_vars):
            abstract_state = list(np.array(self.state_decodings[state])[abstraction])
            for v in range(len(self.params.size_state_vars)):
                if v not in abstraction:
                    abstract_state.insert(v, -1)
            abstract_state = tuple(abstract_state)
            if abstract_state in self.values_for:
                self.values_for[abstract_state].append(self.Q_table[state])
                if len(self.values_for[abstract_state]) > 15:
                    dist_action = np.argmax(np.sum(self.values_for[abstract_state][-15:], axis=0))
                    # print(self.greedy_action(state), dist_action)
                    action = dist_action
                else:
                    action = self.greedy_action(state)
            else:
                self.values_for[abstract_state] = [self.Q_table[state]]
                action = self.greedy_action(state)
        else:
            action = self.greedy_action(state)

        return action

    def update_VA(self, state, action, reward, next_state, done):

        td_error = reward + (not done) * self.params.DISCOUNT * max(self.Q_table[next_state]) - self.Q_table[state][action]
        self.Q_table[state][action] += self.params.ALPHA * td_error

    def do_step(self):
        state = self.current_state
        action = self.legacy_action(state)
        next_state, reward, done = self.step(action)
        self.update_VA(state, action, reward, next_state, done)
        self.current_state = next_state
        if self.steps > self.max_steps:
            done = True
        return reward, done
