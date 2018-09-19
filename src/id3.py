import numpy as np
import copy
from anytree import Node
from FeatureData import FeatureData
from Constants import LESS_THAN_OPERATOR

from util import ENTROPY


def find_best_split(X: np.array, y: np.array, feature_names=None):
    """Find the best split-feature and split-criteria"""
    xc = copy.deepcopy(X)
    best_split = None

    """Uses FeatureData-object to calculate the optimal split for every feature"""
    for idx, feature in enumerate(xc.T):
        unique, count = np.unique(feature, return_counts=True)
        feature_data = FeatureData(idx, feature, y, unique, count)
        split_candidate = feature_data.find_best_split()
        if idx == 0:
            best_split = split_candidate
        elif best_split.information_gain < split_candidate.information_gain:
            best_split = split_candidate

    return best_split


def do_split(data, target, split):
    data_c = copy.deepcopy(data)
    right_data = []
    left_data = []
    right_target = []
    left_target = []
    for idx, x in enumerate(data_c):
        if x[split.column_index] < split.split_value:
            right_data.append(x)
            right_target.append(target[idx])
        else:
            left_data.append(x)
            left_target.append(target[idx])

    right_data = np.array(right_data)
    left_data = np.array(left_data)
    right_target = np.array(right_target)
    left_target = np.array(left_target)

    return right_data, left_data, right_target, left_target


def build_tree(tree, X, y):
    best_split = find_best_split(X, y)
    tree.name = best_split.get_split_name()
    tree.split_operator = best_split.split_operator
    tree.split_value = best_split.split_value
    tree.split_feature_index = best_split.column_index
    tree.prediction = best_split.pure_value

    if best_split.information_gain > 0:
        right_data, left_data, right_target, left_target = do_split(X, y, best_split)
        right_child = Node("Child1", parent=tree)
        left_child = Node("left", parent=tree)
        tree.left = left_child
        tree.right = right_child
        build_tree(right_child, right_data, right_target)
        build_tree(left_child, left_data, left_target)


def learn(X: np.array, y: np.array, feature_names=None, target_name=None, impurity_measure=ENTROPY):
    use = [X, y, feature_names, target_name, impurity_measure]
    root = Node("root")
    build_tree(root, X, y)
    return root


def predict(x, tree, feature_names=None):
    """Predicts the label of a given feature-vector x"""
    if tree.is_leaf:
        prediction = tree.prediction
        return prediction
    else:
        feature_value = x[tree.split_feature_index]
        split_value = tree.split_value
        op = tree.split_operator

        if op == LESS_THAN_OPERATOR:
            print("--")
            print(tree.left.split_value)
