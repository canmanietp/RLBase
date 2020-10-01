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

        self.learn_tree_steps = 15000  # 55000 for taxilarge, # 15000 for taxi, # 50000 for taxifuel
        # self.sample_k = 5000
        self.tree_rules = None
        self.rule_actions = None
        self.rule_quality = None

        self.state_rule_mapping = {}

        self.d = 5  # 10 for taxilarge
        self.rule_limit = 5

        self.next_action = None
        self.rules_Q_table = []

        self.RMAX = 20

    def e_greedy_tree_action(self, state):
        reward, done = None, None

        if self.tree_rules is None:
            return self.e_greedy_action(state), reward, done

        if state not in self.state_rule_mapping:
            self.add_to_mapping(state)

        rules = self.state_rule_mapping[state]

        if rules is not None:
            if np.random.uniform(0, 1) < self.params.EPSILON:
                action = self.random_action()
            else:
                if np.random.uniform(0, 1) < 10 / (1 + np.sum(self.sa_visits[state])):
                    rules_values = [self.rules_Q_table[ro] for ro in rules]
                    max_rule = np.argmax(rules_values)
                    action = self.rule_actions[max_rule]
                else:
                    action = np.argmax(self.Q_table[state])

            next_state, reward, done = self.step(action)

            for r in rules:
                if self.rule_actions[r] == action:
                    td_error = reward + (not done) * self.params.DISCOUNT * max(self.Q_table[next_state]) - self.rules_Q_table[r]
                    self.rules_Q_table[r] = self.rules_Q_table[r] + self.params.ALPHA * td_error

            self.update(state, action, reward, next_state, done)
            self.current_state = next_state

            return action, reward, done
        else:
            return self.e_greedy_action(state), reward, done

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
        # return self.greedy_action(state)
        # return self.greedy_action(state)

    def add_to_mapping(self, state):
        s_vars = list(self.env.decode(state))
        any_valid_rules = False
        action_boosting = np.zeros(self.action_space)

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
                if action_boosting[self.rule_actions[ir]] == 0 and np.sum(self.sa_visits[state]) == 0:
                    print(rules)
                    action_boosting[self.rule_actions[ir]] = (1 * self.RMAX) # / (1 + np.sum(self.sa_visits[state]))

                # if state not in self.expanded_Q_table:
                #     self.expanded_Q_table[state] = self.Q_table[state]
                #     self.expanded_Q_table[state] = np.append(self.expanded_Q_table[state], [0])
                # else:
                #     self.expanded_Q_table[state] = np.append(self.expanded_Q_table[state], [0])
                if state not in self.state_rule_mapping:
                    self.state_rule_mapping[state] = [ir]
                else:
                    self.state_rule_mapping[state].append(ir)

        print("a", self.Q_table[state], action_boosting)
        self.Q_table[state] = self.Q_table[state] + action_boosting
        print(list(self.env.decode(state)), "b", self.Q_table[state])

        if not any_valid_rules:
            self.state_rule_mapping[state] = None

    def reset_tree(self):
        self.tree_rules = None
        self.rule_actions = None
        self.state_rule_mapping = {}

        # distribution according to the visits to the unique states
        # self.unique_state_history
        # visits = [np.sum(self.sa_visits[st]) for st in range(self.observation_space)]
        # prob_visits = visits / np.sum(visits)

        # randomly sample k visited states
        # samples = np.random.choice(self.unique_state_history, self.sample_k, p=prob_visits)

        best_act_save = []

        for s in range(self.observation_space):
            # may need to exclude termination states
            vars = list(self.env.decode(s))
            if ('Taxi' in str(self.env) and vars[2] != vars[3]) or ('Taxi' not in str(self.env)):
                if np.sum(self.sa_visits[s]) > 0:
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
            if r[0] < self.rule_limit and -1 < r[1] < self.action_space:
                filtered_rules.append(r[2:])
                filtered_actions.append(r[1])
                # print(filtered_rules[-1])
                # print(filtered_actions[-1])

        self.tree_rules = filtered_rules
        self.rule_actions = filtered_actions
        self.rule_quality = [[] for ra in self.rule_actions]
        self.rules_Q_table = np.zeros(len(self.tree_rules))

    def do_step(self):
        state = self.current_state
        reward, done = None, None

        # if self.tree_rules is not None:
            # action, reward, done = self.e_greedy_tree_action(state)
        # else:
        #         action = self.e_greedy_action(state)

        if state not in self.state_rule_mapping and self.tree_rules is not None:
            self.add_to_mapping(state)
        action = self.e_greedy_action(state)

        if reward is None:
            next_state, reward, done = self.step(action)
            self.update(state, action, reward, next_state, done)
            self.current_state = next_state

        if state not in self.unique_state_history:
            self.unique_state_history.append(state)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True

        self.steps += 1
        if self.steps > 0 and self.steps % self.learn_tree_steps == 0 and self.tree_rules is None:
            self.reset_tree()
            print("--------------------------------------------------------------------------------------planting tree")
        return reward, done