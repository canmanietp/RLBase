import math

from agents.Q import QAgent
from agents.singlestate import SingleStateAgent
import numpy as np
import copy


class LOARA_K_Agent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'LOARA_known'
        self.params = copy.copy(params)

        self.state_decodings = self.sweep_state_decodings()
        self.bandit_list = []

        self.abs_states_abstate_map = [[([], []) for ss in range(int(np.prod(np.array(self.params.size_state_vars)[a])))] for a in self.params.sub_spaces]
        self.state_bandit_map, self.abs_state_bandit_map = self.init_bandits()
        self.next_bandit, self.next_action = None, None

        self.sb_visits = np.zeros([self.observation_space, len(self.params.sub_spaces)])
        self.reuse_params = [0.8 for os in range(self.observation_space)]

        self.abs_states_abstates_dicts = [dict() for ab in self.params.sub_spaces]
        for ia, ab in enumerate(self.abs_states_abstate_map):
            for keys, value in ab:
                for key in keys:
                    self.abs_states_abstates_dicts[ia][key] = value
                    # print(self.state_decodings[key], value)

    def init_bandits(self):
        state_bandit_map = {}
        abs_state_bandit_map = [{} for ss in self.params.sub_spaces]

        for s in range(self.observation_space):
            s_vars = self.state_decodings[s]
            bandits = []
            for ia, abs_vars in enumerate(self.params.sub_spaces):
                abs_state = self.encode_abs_state(s_vars, abs_vars)
                self.abs_states_abstate_map[ia][abs_state][0].append(s)
                if not self.abs_states_abstate_map[ia][abs_state][1]:
                    self.abs_states_abstate_map[ia][abs_state][1].append(abs_state)
                if abs_state not in abs_state_bandit_map[ia]:
                    abs_state_bandit_map[ia][abs_state] = SingleStateAgent(self.action_space, self.params.ALPHA,
                                                                           self.params.DISCOUNT)
                    self.bandit_list.append(abs_state_bandit_map[ia][abs_state])
                bandits.append(abs_state_bandit_map[ia][abs_state])
            state_bandit_map[s] = bandits
        return state_bandit_map, abs_state_bandit_map

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
            for b in self.bandit_list:
                b.decay(decay_rate)
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def heuristic_bandit_choice(self, state):
        state_vars = self.state_decodings[state]
        if 'TaxiFuel' in str(self.env):
            if state_vars[2] < 4 and state_vars[4] > 12:  # don't have passenger and have enough fuel to get any
                # passenger to any destination
                return self.params.sub_spaces.index([0, 1, 2])
            elif state_vars[2] == 4 and state_vars[4] > 8:  # have passenger and have enough fuel to get to any
                # destination
                return self.params.sub_spaces.index([0, 1, 2, 3])
            elif state_vars[2] == 4 and (state_vars[3] == 0 or state_vars[3] == 2) and state_vars[1] == 0 and state_vars[4] > 4:
                return self.params.sub_spaces.index([0, 1, 2, 3])
            elif state_vars[2] == 4 and (state_vars[3] == 1 or state_vars[3] == 3) and state_vars[1] >= 3 and  state_vars[4] > 4:
                return self.params.sub_spaces.index([0, 1, 2, 3])
            else:
                return len(self.params.sub_spaces) - 1
        elif 'TaxiLarge' in str(self.env):
            # if state_vars[1] == 9 and state_vars[2] == 8 and state_vars[3] == 7:
            #     return self.params.sub_spaces.index([1, 2, 3])
            # elif state_vars[1] == 8 and state_vars[2] == 8 and state_vars[3] == 6:
            #     return self.params.sub_spaces.index([1, 2, 3])
            # elif state_vars[1] == 0 and state_vars[2] == 8 and (state_vars[3] == 0 or state_vars[3] == 2):
            #     return self.params.sub_spaces.index([1, 2, 3])
            # elif state_vars[1] == 4 and state_vars[2] == 8 and (state_vars[3] == 1 or state_vars[3] == 3):
            #     return self.params.sub_spaces.index([1, 2, 3])
            if state_vars[2] < 8:
                return self.params.sub_spaces.index([0, 1, 2])
            else:
                return len(self.params.sub_spaces) - 1
        elif 'Taxi' in str(self.env):
            # if state_vars[0] == 4 and state_vars[1] == 1 or state_vars[0] == 4 and state_vars[1] == 2:
            #     return self.params.sub_spaces.index([2])
            # if state_vars[0] == 3 and state_vars[2] == 4 and state_vars[3] == 0:
            #     return self.params.sub_spaces.index([0, 2, 3])
            # elif state_vars[1] == 4 and state_vars[2] == 4 and state_vars[3] != 1:
            #     return self.params.sub_spaces.index([1, 2, 3])
            if state_vars[2] < 4:
                return self.params.sub_spaces.index([0, 1, 2])
            else:
                return len(self.params.sub_spaces) - 1
        elif 'CoffeeMail' in str(self.env):
            # [0, 1, 2, 4, 6], [0, 1, 3, 5, 7],
            if state_vars[4] == 1 or state_vars[5] == 1:
                # if state_vars[5] != 1:
                #     return self.params.sub_spaces.index([0, 1, 2, 4])
                # if state_vars[4] != 1:
                #     return self.params.sub_spaces.index([0, 1, 3, 5])
                return self.params.sub_spaces.index([0, 1, 2, 3, 4, 5])
            elif state_vars[6] == 1 or state_vars[7] == 1:
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

    def e_greedy_bandit_action(self, state):
        if self.next_bandit is None:
            bandit_index = self.heuristic_bandit_choice(state)
        else:
            bandit_index = self.next_bandit
        if self.next_action is None:
            action = self.state_bandit_map[state][-1].e_greedy_action(self.params.EPSILON)
        else:
            action = self.next_action
        return bandit_index, action

    def calc_value_abstract_state(self, state, bandit_index):
        # For all states that make up the abstract state, normalize and sum their values
        ab_state = self.encode_abs_state(self.state_decodings[state], self.params.sub_spaces[bandit_index])
        merged_states = []
        for s, ab_s in self.abs_states_abstates_dicts[bandit_index].items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if ab_s[0] == ab_state and np.sum(self.sa_visits[s]) > 0:
                merged_states.append(s)

        norm_sum = np.zeros(self.action_space)
        merged_values = []
        merged_visits = []
        one_hots = []

        for ms in merged_states:
            values = self.state_bandit_map[ms][-1].Q_table
            visits = self.state_bandit_map[ms][-1].action_visits
            one_hot = []
            for v in values:
                if v == max(values) and all(p for p in values if p == 0):
                    one_hot.append(1)
                else:
                    one_hot.append(0)

            one_hots.append(one_hot)
            merged_visits.append(np.sum(visits))
            merged_values.append(values)

            # e_x = np.exp(x - np.max(x))
            # e_x / e_x.sum(axis=0)

            # if np.sum(visits) != 0:
            #     norm = [(v - min(values)) / (max(values) - min(values)) for v in values]
            # #     merged_values += values
            #     merged_values.append(values)
            #     merged_visits.append(visits)
            #     norm_sum += norm

        if not merged_values or np.sum(one_hots) == 0:
            return self.state_bandit_map[state][-1].Q_table
        return merged_values[int(np.argmax(merged_visits))]

    def update_LIA(self, state, bandit_index, action, reward, next_state, done):
        self.next_bandit = self.heuristic_bandit_choice(next_state)
        if np.random.uniform(0, 1) < self.params.EPSILON:
            self.next_action = self.state_bandit_map[next_state][-1].random_action()
        else:
            test = self.calc_value_abstract_state(next_state, self.next_bandit)
            self.next_action = np.argmax(test)  # self.state_bandit_map[next_state][-1].e_greedy_action(self.params.EPSILON)
        # print(self.state_decodings[state], bandit_index, action, reward, done, self.state_decodings[next_state], self.next_bandit, self.next_action, test)
        # if self.steps > 800:
        #     quit()
        # alt_val = self.state_bandit_map[next_state][self.next_bandit].Q_table[self.next_action]
        # val = max(self.calc_value_abstract_state(next_state, self.next_bandit))

        # if bandit_index != len(self.params.sub_spaces) - 1:
        #     self.state_bandit_map[state][bandit_index].update(action, reward + self.params.DISCOUNT * (not done) * alt_val)

        self.state_bandit_map[state][-1].update(action, reward + self.params.DISCOUNT * (not done) * max(self.state_bandit_map[next_state][-1].Q_table))

    def do_step(self):
        state = self.current_state
        bandit_index, action = self.e_greedy_bandit_action(state)
        next_state, reward, done = self.step(action)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update_LIA(state, bandit_index, action, reward, next_state, done)
        self.current_state = next_state
        self.steps += 1
        if done:
            self.next_bandit = None
            self.next_action = None
        return reward, done