from agents.Q import QAgent
from agents.singlestate import SingleStateAgent
import numpy as np
import pickle
import copy


class MergeTree_Agent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'MergeTree'
        self.params = copy.copy(params)

        infile = open('helpers/pickled_tree_rules_{}'.format(str(env)), 'rb')
        self.tree_rules = pickle.load(infile)
        self.abstract_states = [SingleStateAgent(self.action_space, self.params.ALPHA, self.params.DISCOUNT) for tr in self.tree_rules]

        if 'TaxiEnv' in str(env):
            self.var_names = ['row', 'col', 'pass', 'dest']

        self.state_decodings = self.sweep_state_decodings()
        self.abstraction_mapping = self.init_abstraction_mapping()

    def sweep_state_decodings(self):
        st_vars_lookup = []
        for s in range(self.observation_space):
            st_vars_lookup.append(list(self.env.decode(s)))
        return st_vars_lookup

    def init_abstraction_mapping(self):
        state_abstraction_map = [[False for tr in self.tree_rules] for os in range(self.observation_space)]
        for s in range(self.observation_space):
            s_vars = self.state_decodings[s]
            if s_vars[2] != s_vars[3]:
                for ir, rules in enumerate(self.tree_rules):
                    valid = True
                    for r in rules:
                        if r[1] == 'r':
                            if s_vars[self.var_names.index(r[3])] <= r[2]:
                                valid = False
                        elif r[1] == 'l':
                            if s_vars[self.var_names.index(r[3])] > r[2]:
                                valid = False
                        else:
                            print("INVALID TREE RULE")
                            quit()
                    if valid:
                        state_abstraction_map[s][ir] = True
            # if all(not check for check in state_abstraction_map[s]):
            #     pass
            #     # print(s_vars)
            # else:
            #     print(np.argmax(state_abstraction_map[s]), s_vars, self.tree_rules[np.argmax(state_abstraction_map[s])])
        return state_abstraction_map

    def decay(self, decay_rate):
        if self.params.ALPHA > self.params.ALPHA_MIN:
            self.params.ALPHA *= decay_rate
            for b in self.abstract_states:
                b.decay(decay_rate)
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def state_value(self, state):
        merged_state = [ix for ix, x in enumerate(self.abstraction_mapping[state]) if x]
        if not merged_state:
            # print("actual values", self.Q_table[state])
            return max(self.Q_table[state])
        merged_state = merged_state[0]
        values = [max(self.abstract_states[merged_state].Q_table), max(self.Q_table[state])]
        print(self.state_decodings[state], "actual values", self.Q_table[state], "merged values", self.abstract_states[merged_state].Q_table)
        return max(values)

    def e_greedy_tree_action(self, state):
        merged_state = [ix for ix, x in enumerate(self.abstraction_mapping[state]) if x]
        if not merged_state:
            return self.e_greedy_action(state)
        if np.random.uniform(0, 1) < self.params.EPSILON:
            merged_state = np.random.randint(len(self.tree_rules))
        else:
            merged_state = merged_state[0]
        values = [max(self.abstract_states[merged_state].Q_table), max(self.Q_table[state])]
        if np.max(values) == values[-1]:
            return self.e_greedy_action(state)
        action = self.abstract_states[merged_state].greedy_action()
        return action

    def update_LIA(self, state, action, reward, next_state, done):
        val = self.state_value(next_state)
        next_val = (not done) * val
        merged_state = [ix for ix, x in enumerate(self.abstraction_mapping[state]) if x]
        if merged_state:
            merged_state = merged_state[0]
            self.abstract_states[merged_state].update(action, reward + self.params.DISCOUNT * (not done) * max(self.Q_table[next_state]))
            print(self.state_decodings[state], self.tree_rules[merged_state], action, reward, done)
        self.Q_table[state][action] += self.params.ALPHA * (reward + self.params.DISCOUNT * (not done) * max(self.Q_table[next_state]) - self.Q_table[state][action])

    def do_step(self):
        state = self.current_state
        action = self.e_greedy_tree_action(state)
        next_state, reward, done = self.step(action)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update_LIA(state, action, reward, next_state, done)
        self.current_state = next_state
        self.steps += 1
        return reward, done