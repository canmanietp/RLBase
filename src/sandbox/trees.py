import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import pickle
from helpers.functions import get_lineage

env = 'TaxiEnv' # WarehouseEnv'

optimal_actions = genfromtxt('Q_samples_{}.csv'.format('<{} instance>'.format(env)), delimiter=',')
FEATURE_NAMES = ["row", "col", "pass", "dest"]  # , "fuel" #  # ['loc', '0', '1', '2', '3'] #

X = pd.DataFrame(optimal_actions[:, 0:len(FEATURE_NAMES)], columns=FEATURE_NAMES)

y = optimal_actions[:, len(FEATURE_NAMES)]

# print(X)
# print(y)

model = DecisionTreeClassifier()
model.fit(X,y)
export_graphviz(model, 'tree.dot', feature_names=FEATURE_NAMES)

decision_rules = get_lineage(model, FEATURE_NAMES)
filtered_rules = []

for r in decision_rules:
    if r[0] < 5:
        filtered_rules.append(r[2:])
        print(r)

filename = 'pickled_tree_rules_{}'.format('<{} instance>'.format(env))
outfile = open(filename,'wb')
pickle.dump(filtered_rules, outfile)
outfile.close()

mapping = [[] for r in filtered_rules]

# print(filtered_rules)

for isv, state_vars in enumerate(optimal_actions[:, 0:len(FEATURE_NAMES)]):
    for ir, rules in enumerate(filtered_rules):  # [[(0, 'l', 4, 'pass')]]
        valid = True
        for r in rules:
            if r[1] == 'r':
                if state_vars[FEATURE_NAMES.index(r[3])] < r[2]:
                    valid = False
            elif r[1] == 'l':
                if state_vars[FEATURE_NAMES.index(r[3])] >= r[2]:
                    valid = False
        if valid and list(state_vars) not in mapping[ir]:
            mapping[ir].append(list(state_vars))

# print(mapping)

filename = 'state_mapping_{}'.format('<{} instance>'.format(env))
outfile = open(filename,'wb')
pickle.dump(mapping, outfile)
outfile.close()