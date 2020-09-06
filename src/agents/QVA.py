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

        infile = open('helpers/state_mapping_{}'.format(str(env)), 'rb')
        self.state_mapping = pickle.load(infile)

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

    def heuristic_abstraction_choice(self, state):
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
            if state_vars[1] == 9 and state_vars[2] == 8 and state_vars[3] == 7:
                return self.params.sub_spaces.index([1, 2, 3])
            elif state_vars[1] == 8 and state_vars[2] == 8 and state_vars[3] == 6:
                return self.params.sub_spaces.index([1, 2, 3])
            elif state_vars[1] == 0 and state_vars[2] == 8 and (state_vars[3] == 0 or state_vars[3] == 2):
                return self.params.sub_spaces.index([1, 2, 3])
            elif state_vars[1] == 4 and state_vars[2] == 8 and (state_vars[3] == 1 or state_vars[3] == 3):
                return self.params.sub_spaces.index([1, 2, 3])
            if state_vars[2] < 8:
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
                # if state_vars[5] == 0:
                #     return self.params.sub_spaces.index([0, 1, 2, 4])
                # if state_vars[4] == 0:
                #     return self.params.sub_spaces.index([0, 1, 3, 5])
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

    def legacy_action(self, state):
        ab_index = self.heuristic_abstraction_choice(state)
        action = self.e_greedy_action(state)

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
            return 0, action, 0
        else:
            no = 0
            all = 0

            for im, mv in enumerate(merge_values):
                if self.sa_visits[update_states[im]][action] > 0:
                    if np.argmax(mv) != action:
                        no += 1
                    all += 1

            percent_agreeance = no / all if all > 0 else 0

            return percent_agreeance, action, np.sum([mv[action] for mv in merge_visits])

    def update_VA(self, state, ab_diff, test, action, reward, next_state, done):
        # self.meta_agent.update(state, ab_index, reward, next_state, done)
        # self.update(state, action, reward, next_state, done)

        beta = max([1, 3. / (1 + self.sa_visits[state][action])])
        # print(self.state_decodings[state], action, beta, ab_diff, beta * ab_diff)

        adjusted_reward = reward - (abs(reward) * ab_diff * beta)

        td_error = adjusted_reward + (not done) * self.params.DISCOUNT * max(self.Q_table[next_state]) - self.Q_table[state][action]

        # adjusted_td_error = td_error - beta * ab_diff
        # print(td_error, ab_diff, adjusted_td_error)

        # print(self.state_decodings[state], action, td_error, adjusted_td_error)
        self.Q_table[state][action] += self.params.ALPHA * td_error

    def do_step(self):
        state = self.current_state
        ab_diff, action, test = self.legacy_action(state)
        next_state, reward, done = self.step(action)
        self.update_VA(state, ab_diff, test, action, reward, next_state, done)
        self.current_state = next_state
        if self.steps > self.max_steps:
            done = True
        return reward, done
