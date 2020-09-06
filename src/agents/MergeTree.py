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
        self.abstract_states = [SingleStateAgent(self.action_space, self.params.ALPHA, self.params.DISCOUNT) for tr in
                                self.tree_rules]
        self.leaves = [[] for tr in self.tree_rules]

        if 'TaxiFuel' in str(env):
            self.var_names = ['row', 'col', 'pass', 'dest', 'fuel']
        elif 'Taxi' in str(env):
            self.var_names = ['row', 'col', 'pass', 'dest']
        elif 'Warehouse' in str(env):
            self.var_names = ['loc']
            for z in range(len(self.env.locs)):
                self.var_names.append('{}'.format(z))

        self.state_decodings = self.sweep_state_decodings()
        self.abstraction_mapping = self.init_abstraction_mapping()

        self.abst_chooser = np.zeros([self.observation_space, 2])

        self.next_action = None
        self.next_action_chooser = None

        self.raw_Q_table = np.zeros([self.observation_space, self.action_space])
        self.beta0, self.beta1 = 1., 0.

    def sweep_state_decodings(self):
        st_vars_lookup = []
        for s in range(self.observation_space):
            st_vars_lookup.append(list(self.env.decode(s)))
        return st_vars_lookup

    def init_abstraction_mapping(self):
        state_abstraction_map = [[False for tr in self.tree_rules] for os in range(self.observation_space)]
        for s in range(self.observation_space):
            s_vars = self.state_decodings[s]
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
                if valid:  # and s_vars[2] != s_vars[3]:  ## Hack to deal with terminal states where optimal action looks like 0 but isn't
                    state_abstraction_map[s][ir] = True
                    self.leaves[ir].append(s)
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
            return max(self.Q_table[state])
        values = [max(self.abstract_states[m].Q_table) for m in merged_state]
        values.append(max(self.Q_table[state]))
        # print(self.state_decodings[state], "actual values", self.Q_table[state], "merged values", self.abstract_states[merged_state].Q_table)
        return max(values)

    def e_greedy_tree_action(self, state):
        if self.next_action is not None:
            return self.next_action, self.next_action_chooser

        if np.random.uniform(0, 1) < self.params.EPSILON:
            return self.random_action()

        merged_state = [ix for ix, x in enumerate(self.abstraction_mapping[state]) if x]

        if not merged_state:
            return np.argmax(self.raw_Q_table[state])

        highest_val = float("-inf")
        action = -1
        for m in merged_state:
            if max(self.abstract_states[m].Q_table) > highest_val:
                highest_val = max(self.abstract_states[m].Q_table)
                action = np.argmax(self.abstract_states[m].Q_table)

        # action = self.abstract_states[merged_state].e_greedy_action(self.params.EPSILON)
        if max(self.raw_Q_table[state]) > highest_val:
            return np.argmax(self.raw_Q_table[state])

        return action

    def e_greedy_weighted_action(self, state):
        if np.random.uniform(0, 1) < self.params.EPSILON:
            return self.random_action()

        self.beta0 = min([1., 3. / (1 + np.sum(self.sa_visits[state]))])
        self.beta1 = 1. - self.beta0
        new_Q = np.zeros(self.action_space)
        merged_state = [ix for ix, x in enumerate(self.abstraction_mapping[state]) if x]

        merged_state = [merged_state[0]] if merged_state else merged_state

        for a in range(self.action_space):
            new_Q[a] = (self.beta0 * max(
                [self.abstract_states[m].Q_table[a] for m in merged_state]) if merged_state else 0.) + (
                                   self.beta1 * self.raw_Q_table[state][a])

        return np.argmax(new_Q)

    def update_LIA(self, state, action, reward, next_state, done):
        merged_state = [ix for ix, x in enumerate(self.abstraction_mapping[state]) if x]
        next_val = self.state_value(next_state)

        for m in merged_state:
            self.abstract_states[m].update(action, reward + self.params.DISCOUNT * (not done) * next_val)

        self.raw_Q_table[state][action] += self.params.ALPHA * (
                    reward + self.params.DISCOUNT * (not done) * max(self.raw_Q_table[next_state]) -
                    self.raw_Q_table[state][action])

    def do_step(self):
        state = self.current_state
        action = self.e_greedy_weighted_action(state)
        next_state, reward, done = self.step(action)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update_LIA(state, action, reward, next_state, done)
        self.current_state = next_state
        self.steps += 1
        if done:
            self.next_action, self.next_action_chooser = None, None
        return reward, done
