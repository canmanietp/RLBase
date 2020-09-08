import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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
