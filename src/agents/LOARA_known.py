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

        self.state_bandit_map, self.abs_state_bandit_map, self.bandit_list = self.init_bandits()
        self.next_bandit, self.next_action = None, None

    def init_bandits(self):
        state_bandit_map = {}
        abs_state_bandit_map = [{} for ss in self.params.sub_spaces]
        bandit_list = []
        for s in range(self.observation_space):
            s_vars = self.state_decodings[s]
            bandits = []
            for ia, abs_vars in enumerate(self.params.sub_spaces):
                abs_state = self.encode_abs_state(s_vars, abs_vars)
                if abs_state not in abs_state_bandit_map[ia]:
                    abs_state_bandit_map[ia][abs_state] = SingleStateAgent(self.action_space, self.params.ALPHA, self.params.DISCOUNT)
                    bandit_list.append(abs_state_bandit_map[ia][abs_state])
                    bandits.append(abs_state_bandit_map[ia][abs_state])
                else:
                    bandits.append(abs_state_bandit_map[ia][abs_state])
            state_bandit_map[s] = bandits
        return state_bandit_map, abs_state_bandit_map, bandit_list

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
            # if state_vars[0] < len(self.env.locs) - 1 and state_vars[state_vars[0] + 1] > 0:
            #     # print(state_vars, [0, state_vars[0] + 1])
            #     return self.params.sub_spaces.index([0, state_vars[0] + 1])
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
            action = self.state_bandit_map[state][bandit_index].e_greedy_action(self.params.EPSILON)
        else:
            action = self.next_action
        return bandit_index, action

    def update_LIA(self, state, bandit_index, action, reward, next_state, done):
        self.next_bandit = self.heuristic_bandit_choice(next_state)
        self.next_action = self.state_bandit_map[next_state][self.next_bandit].e_greedy_action(self.params.EPSILON)
        # action_values = [b.Q_table[self.next_action] for ib, b in enumerate(self.state_bandit_map[next_state])]
        val = max(self.state_bandit_map[next_state][self.next_bandit].Q_table)

        # if 'Taxi' in str(self.env):
        #     if self.params.sub_spaces[bandit_index] == [0, 1, 2]:
        #         state_vars = self.state_decodings[state]
        #         reward = -1
        #
        #         if action == 4:  # pickup
        #             if not (state_vars[2] < len(self.env.locs) and (state_vars[0], state_vars[1]) == self.env.locs[state_vars[2]]):
        #                 reward = -10
        #         elif action == 5:  # dropoff
        #             if not ((state_vars[0], state_vars[1]) in self.env.locs) and state_vars[2] == 4:
        #                 reward = -10
        # if 'Warehouse' in str(self.env):
        #     if 0 not in self.params.sub_spaces[bandit_index]:
        #         reward = -1
        #         state_vars = self.state_decodings[state]
        #         if action >= len(self.env.locs) - 1 and not all(nr == 0 for nr in state_vars[1:]):
        #             reward -= 11

        # print(list(self.env.decode(state)), bandit_index, action, reward, list(self.env.decode(next_state)), done)

        self.state_bandit_map[state][bandit_index].update(action, reward + self.params.DISCOUNT * (not done) * val)

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