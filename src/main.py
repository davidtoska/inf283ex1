# import numpy as np
import data_prep
from anytree import RenderTree

from id3 import learn, predict, ENTROPY


def print_tree(root):
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))


# Load iris data

X, y, feature_names, target_name = data_prep.get_iris_data()

id3 = learn(X, y, target_name=target_name, impurity_measure=ENTROPY)

print_tree(id3)

testX = X[0]
testY = y[0]
p = predict(testX, id3)

print(p)
"""
#X_mushroom, y_mushroom, feature_names_mushroom, target_name_mushroom = data_prep.get_mushroom_data()
"""


def main():
    print("Main method is running")
