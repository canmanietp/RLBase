import random

from agents.Q import QAgent
from agents.singlestate import SingleStateAgent
import numpy as np
import copy


class LOARA_UK_Agent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'LOARA_unknown'
        self.params = copy.copy(params)

        self.state_decodings = self.sweep_state_decodings()
        self.bandit_list = []
        self.ab_states_abstate_map = [[([], []) for ss in range(int(np.prod(np.array(self.params.size_state_vars)[a])))] for a in self.params.sub_spaces]
        self.state_bandit_map, self.abs_state_bandit_map = self.init_bandits()
        self.next_bandit, self.next_action = None, None

        self.sb_visits = np.zeros([self.observation_space, len(self.params.sub_spaces)])

        # state_abstates_dicts = [dict() for ab in self.params.sub_spaces]
        # for ia, ab in enumerate(self.ab_states_abstate_map):
        #     for keys, value in ab:
        #         for key in keys:
        #             state_abstates_dicts[ia][key] = value

    def init_bandits(self):
        state_bandit_map = {}
        abs_state_bandit_map = [{} for ss in self.params.sub_spaces]

        for s in range(self.observation_space):
            s_vars = self.state_decodings[s]
            bandits = []
            for ia, abs_vars in enumerate(self.params.sub_spaces):
                abs_state = self.encode_abs_state(s_vars, abs_vars)
                self.ab_states_abstate_map[ia][abs_state][0].append(s)
                if not self.ab_states_abstate_map[ia][abs_state][1]:
                    self.ab_states_abstate_map[ia][abs_state][1].append(abs_state)
                if abs_state not in abs_state_bandit_map[ia]:
                    abs_state_bandit_map[ia][abs_state] = SingleStateAgent(self.action_space, self.params.ALPHA,
                                                                           self.params.DISCOUNT)
                    self.bandit_list.append(abs_state_bandit_map[ia][abs_state])
                    bandits.append(abs_state_bandit_map[ia][abs_state])
                else:
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

    def e_greedy_bandit(self, state):
        if random.uniform(0, 1) < self.params.EPSILON:
            self.next_action = self.state_bandit_map[state][-1].random_action()
            return np.random.randint(len(self.params.sub_spaces) - 1), max(self.Q_table[state])
        bandit_index = int(np.argmax(self.Q_table[state]))
        self.next_action = self.state_bandit_map[state][bandit_index].random_action()
        return bandit_index, max(self.Q_table[state])

    def smart_bandit_choice(self, state, eps=None):
        values = [max(b.Q_table) for ib, b in enumerate(self.state_bandit_map[state])]
        if np.random.uniform(0, 1) < (eps if eps else self.params.EPSILON):
            bandit_index = np.random.randint(len(self.params.sub_spaces))
        else:
            bandit_index = int(np.argmax(values))
        self.next_action = self.state_bandit_map[state][bandit_index].greedy_action()  # .e_greedy_action(eps if eps else self.params.EPSILON)
        return bandit_index, max(values)

    def e_greedy_bandit_action(self, state):
        if self.next_bandit is None:
            bandit_index = np.argmax([max(b.Q_table) for b in self.state_bandit_map[state]])
        else:
            bandit_index = self.next_bandit
        if self.next_action is None:
            action = self.state_bandit_map[state][bandit_index].e_greedy_action(self.params.EPSILON)
        else:
            action = self.next_action
        return bandit_index, action

    def update_LIA(self, state, bandit_index, action, reward, next_state, done):
        self.next_bandit, self.next_action = None, None
        self.next_bandit, val = self.smart_bandit_choice(next_state)
        next_val = (not done) * val
        # print(self.state_decodings[state], bandit_index, action, reward, done, [max(b.Q_table) for b in self.state_bandit_map[state]])
        for ib, b in enumerate(self.state_bandit_map[state]):
            if set(self.params.sub_spaces[bandit_index]).issubset(set(self.params.sub_spaces[ib])):
                self.state_bandit_map[state][ib].update(action, reward + self.params.DISCOUNT * next_val)
        # if bandit_index != len(self.params.sub_spaces) - 1:
            # self.state_bandit_map[state][-1].update(action, reward + self.params.DISCOUNT * next_val)
        # self.state_bandit_map[state][-1].update(action, reward + self.params.DISCOUNT * (not done) * max(self.state_bandit_map[next_state][-1].Q_table))

    def do_step(self):
        state = self.current_state
        bandit_index, action = self.e_greedy_bandit_action(state)
        next_state, reward, done = self.step(action)
        self.sb_visits[state][bandit_index] += 1
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update_LIA(state, bandit_index, action, reward, next_state, done)
        self.steps += 1
        if done:
            self.next_bandit, self.next_action = None, None
        return reward, done
