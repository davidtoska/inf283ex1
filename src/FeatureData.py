import numpy as np
import copy
from util import calc_information_gain
from SplitData import SplitData


class FeatureData:
    """Utility class to encapsulate all calculations on a single feature-vector"""

    def __init__(self, column_index: int, feature: np.array, targets: np.array,
                 unique_values: np.array, value_count: np.array):
        self.feature = copy.deepcopy(feature)
        self.targets = copy.deepcopy(targets)
        self.column_index = column_index
        self.unique_values: np.array = unique_values
        self.value_count: np.array = value_count
        self.value_type = unique_values.dtype
        self.unique_targets = np.unique(targets)
        self.number_of_unique_targets = len(np.unique(targets))
        self.split = None
        self.information_gain: float = -1.0

    def get_numeric_split_candidates(self):
        """Returns a list of all possible split candidates"""
        candidates = []
        for i in range(1, len(self.unique_values)):
            split = (self.unique_values[i - 1] + self.unique_values[i]) / 2
            candidates.append(split)
        return candidates

    def split_on_numeric_value(self, y: np.array, split_value: float):
        """Splits both feature and targets in new arrays"""
        right_x = []
        left_x = []
        right_y = []
        left_y = []
        for idx, xi in enumerate(self.feature):
            if xi < split_value:
                right_x.append(xi)
                right_y.append(y[idx])
            else:
                left_x.append(xi)
                left_y.append(y[idx])
        return right_x, left_x, right_y, left_y

    def split_numeric_feature(self):
        """Calculate the best split of a numeric feature and returns a split-object"""
        candidates = self.get_numeric_split_candidates()
        best_information_gain = -1.0
        best_split_value = -1.0
        for idx, c in enumerate(candidates):
            rx, lx, ry, ly = self.split_on_numeric_value(self.targets, c)
            info_gain = calc_information_gain(self.targets, ry, ly)
            if info_gain > best_information_gain:
                best_information_gain = info_gain
                best_split_value = c

        # Create a container object for the split information
        split = SplitData(self.column_index)
        split.split_value = best_split_value
        split.information_gain = best_information_gain
        split.is_numeric = True
        split.split_operator = "<"
        return split

    def find_best_split(self):
        best_split = SplitData(self.column_index)
        if self.number_of_unique_targets == 1:
            best_split.pure_value = self.unique_targets[0]
            best_split.pure_value_count = len(self.targets)
        elif self.value_type == np.float:
            best_split = self.split_numeric_feature()
        return best_split
