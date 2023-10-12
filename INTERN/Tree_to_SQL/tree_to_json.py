import json
from sklearn.tree import DecisionTreeClassifier


def _convert_tree(tree: DecisionTreeClassifier, node_index: int) -> dict:
    """Return the decision tree node as a dict. Execute recursively"""
    
    is_leaf = tree.children_left[node_index] == -1 and tree.children_right[node_index] == -1
    
    if is_leaf:
        return {"class": int(tree.value[node_index].argmax())}
    else:
        feature_index = int(tree.feature[node_index])
        threshold = float(tree.threshold[node_index])
        left_child_index = tree.children_left[node_index]
        right_child_index = tree.children_right[node_index]
        return {
                "feature_index": feature_index,
                "threshold": round(threshold,4),
                "left":  _convert_tree(tree, left_child_index),
                "right":  _convert_tree(tree, right_child_index)
        }

def convert_tree_to_json(tree: DecisionTreeClassifier) -> str:
    """Return the decision tree as a JSON string"""
    node_index = 0
    dct_tree = _convert_tree(tree.tree_, node_index)
    tree_as_json = json.dumps(dct_tree)
    return tree_as_json
