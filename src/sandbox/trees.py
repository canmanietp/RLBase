import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pickle


def get_lineage(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    results = []

    for child in idx:
        rules = 0
        backward_results = []
        for nx, node in enumerate(recurse(left, right, child)):
            if type(node) == np.int64:
                # print(np.argmax(value[node]))
                # print("# of rules", rules)
                backward_results.append(np.argmax(value[node]))
                backward_results.append(rules)
                results.append(list(reversed(backward_results)))
                backward_results = []
            else:
                dominates = False
                for br in backward_results:
                    if br[3] == node[3]:
                        if br[1] == node[1]:
                            if (br[1] == 'l' and float(node[2]) < float(br[2])) or (br[1] == 'r' and float(node[2]) > float(br[2])):
                                dominates = True
                if not dominates:
                    backward_results.append(node)
                    rules += 1

            # print(node if type(node) == np.int64 else node[0])

    return results


env = 'TaxiEnv' # WarehouseEnv'

optimal_actions = genfromtxt('Q_samples_{}.csv'.format('<{} instance>'.format(env)), delimiter=',')
FEATURE_NAMES = ["row", "col", "pass", "dest"]  # , "fuel" # ['loc', '0', '1', '2', '3'] #

X = pd.DataFrame(optimal_actions[:, 0:len(FEATURE_NAMES)], columns=FEATURE_NAMES)

y = optimal_actions[:, len(FEATURE_NAMES)]

model = DecisionTreeClassifier()
model.fit(X,y)
export_graphviz(model, 'tree.dot', feature_names=FEATURE_NAMES)

decision_rules = get_lineage(model, FEATURE_NAMES)
filtered_rules = []

for r in decision_rules:
    if r[0] <= 4:
        filtered_rules.append(r[2:])
        print(r)

filename = 'pickled_tree_rules_{}'.format('<{} instance>'.format(env))
outfile = open(filename,'wb')
pickle.dump(filtered_rules, outfile)
outfile.close()

mapping = {}

for state_vars in optimal_actions[:, 0:len(FEATURE_NAMES)]:
    for ir, rules in enumerate(filtered_rules):
        valid = True
        for r in rules:
            if r[1] == 'r':
                if state_vars[FEATURE_NAMES.index(r[3])] <= r[2]:
                    valid = False
            elif r[1] == 'l':
                if state_vars[FEATURE_NAMES.index(r[3])] > r[2]:
                    valid = False
        if valid:
            mapping[tuple(state_vars)] = ir

print(mapping)

filename = 'state_mapping_{}'.format('<{} instance>'.format(env))
outfile = open(filename,'wb')
pickle.dump(mapping, outfile)
outfile.close()