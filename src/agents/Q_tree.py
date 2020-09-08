from agents.Q import QAgent
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from helpers.functions import get_lineage


class Q_TreeAgent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'Q Tree'

        if 'TaxiFuel' in str(env):
            self.FEATURE_NAMES = ["row", "col", "pass", "dest", "fuel"]
        elif 'Taxi' in str(env):
            self.FEATURE_NAMES = ["row", "col", "pass", "dest"]
        # ['loc', '0', '1', '2', '3'] #

        self.unique_state_history = []

        self.learn_tree_steps = 10000
        self.sample_k = 2000
        self.tree_rules = None
        self.rule_actions = None
        self.rule_quality = None

        self.state_rule_mapping = {}

        self.d = 10
        self.rule_limit = 5

        self.next_action = None

    def e_greedy_tree_action(self, state):
        if self.tree_rules is None:
            return self.e_greedy_action(state)

        if np.random.uniform(0, 1) < self.params.EPSILON:
            return self.random_action()

        if state not in self.state_rule_mapping:
            self.add_to_mapping(state)

        if len(self.Q_table[state]) == self.action_space:
            return self.greedy_action(state)



        # beta = min([1, self.d/ (1 + np.sum(self.sa_visits[state]))])
        #
        # if np.random.uniform(0, 1) < beta:
        #     if state not in self.state_rule_mapping:
        #         self.add_to_mapping(state)
        #
        #     if self.state_rule_mapping[state] is None:
        #         return self.greedy_action(state)
        #
        #     rule = np.random.choice(self.state_rule_mapping[state])
        #     action = self.rule_actions[rule]
        #     self.rule_quality[rule].append(int(action == np.argmax(self.Q_table[state])))
        #     # quality = np.count_nonzero(np.array(self.rule_quality[rule]) == 1) / len(self.rule_quality[rule]) if self.rule_quality[rule] else -1
        #     # print(list(self.env.decode(state)), quality, self.tree_rules[rule], self.rule_actions[rule])
        #     return action
        return self.greedy_action(state)
        # return self.greedy_action(state)

    def add_to_mapping(self, state):
        s_vars = list(self.env.decode(state))
        any_valid_rules = False
        for ir, rules in enumerate(self.tree_rules):
            valid = True
            for r in rules:
                if r[1] == 'r':
                    if s_vars[self.FEATURE_NAMES.index(r[3])] < r[2]:
                        valid = False
                elif r[1] == 'l':
                    if s_vars[self.FEATURE_NAMES.index(r[3])] >= r[2]:
                        valid = False
                else:
                    print("INVALID TREE RULE")
                    quit()
            if valid:
                any_valid_rules = True
                self.Q_table[state].append(0.)
                if state not in self.state_rule_mapping:
                    self.state_rule_mapping[state] = [ir]
                else:
                    self.state_rule_mapping[state].append(ir)
        if not any_valid_rules:
            self.state_rule_mapping[state] = None

    def reset_tree(self):
        self.tree_rules = None
        self.rule_actions = None
        self.state_rule_mapping = {}

        # distribution according to the visits to the unique states
        # self.unique_state_history
        visits = [np.sum(self.sa_visits[st]) for st in range(self.observation_space)]
        prob_visits = visits / np.sum(visits)

        # randomly sample k visited states
        samples = np.random.choice(range(self.observation_space), self.sample_k, p=prob_visits)

        best_act_save = []

        for s in samples:
            # may need to exclude termination states
            vars = list(self.env.decode(s))
            if ('Taxi' in str(self.env) and vars[2] != vars[3]) or ('Taxi' not in str(self.env)):
                best_act = np.argmax(self.Q_table[s])
                best_act_save.append([*vars, best_act])

        best_act_save = np.array(best_act_save)

        X = pd.DataFrame(best_act_save[:, 0:len(self.FEATURE_NAMES)], columns=self.FEATURE_NAMES)
        y = best_act_save[:, len(self.FEATURE_NAMES)]

        model = DecisionTreeClassifier()
        model.fit(X, y)
        export_graphviz(model, 'tree.dot', feature_names=self.FEATURE_NAMES)

        decision_rules = get_lineage(model, self.FEATURE_NAMES)

        filtered_rules = []
        filtered_actions = []

        for r in decision_rules:
            if r[0] < self.rule_limit:
                filtered_rules.append(r[2:])
                filtered_actions.append(r[1])
                # print(filtered_rules[-1])
                # print(filtered_actions[-1])

        self.tree_rules = filtered_rules
        self.rule_actions = filtered_actions
        self.rule_quality = [[] for ra in self.rule_actions]

    def do_step(self):
        state = self.current_state
        if self.tree_rules is not None:
            action, default = self.e_greedy_tree_action(state)
        else:
            action = self.e_greedy_action(state)
        next_state, reward, done = self.step(action)
        if state not in self.unique_state_history:
            self.unique_state_history.append(state)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        if default:
            self.update(state, action, reward, next_state, done)
        self.current_state = next_state
        self.steps += 1
        if self.steps > 0 and self.steps % self.learn_tree_steps == 0 and self.tree_rules is None:
            self.reset_tree()
            print("--------------------------------------------------------------------------------------planting tree")
        return reward, done