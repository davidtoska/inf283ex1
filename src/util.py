import numpy as np
import copy

ENTROPY = "entropy"
GINI_INDEX = "gini_index"


def is_numeric(v):
    """Check if the value is a numeric value (float or int)"""
    is_int = isinstance(v, int)
    is_float = isinstance(v, float)
    return is_int or is_float


def calc_entropy(arr: np.array):
    """
    Helper method that calculate the entropy of an given numpy array
    :param arr: numpy array
    :return: entropy: numeric value
    """

    entropy = 0
    total_count = len(arr)

    if total_count == 0:
        return entropy
    else:
        unique, counts = np.unique(arr, return_counts=True)
        # Calculate sum of entropy of all unique values
        for n in range(len(unique)):
            entropy -= (counts[n] / total_count) * np.log2((counts[n] / total_count))
    return entropy


def calc_information_gain(y0: np.array, right: np.array, left: np.array, impurity_measure=ENTROPY):
    """
    Calculate the information gained from a given split (left, right). We
    use the weighted average from the two partitions.
    :param y0:
    :param right:
    :param left:
    :return: information gained
    """

    if impurity_measure == ENTROPY:
        # Calculate the entropy
        e0 = calc_entropy(y0)
        e_right = calc_entropy(right)
        e_left = calc_entropy(left)
        # Use the number in each element to calculate their relative weights.
        n_total = len(y0)
        if n_total > 0:
            w_right = len(right) / n_total
            w_left = len(left) / n_total

        return e0 - (w_left * e_left) - (w_right * e_right)

    elif impurity_measure == GINI_INDEX:
        return 1

    def split_on_numeric_value(feature: np.array, target: np.array, split_value: float):
        """Splits both feature and targets into two arrays (left, right) according to split_value"""
        feat = copy.deepcopy(feature)
        targ = copy.deepcopy(target)
        right_x = []
        left_x = []
        right_y = []
        left_y = []
        for idx, xi in enumerate(feat):
            if xi < split_value:
                right_x.append(xi)
                right_y.append(targ[idx])
            else:
                left_x.append(xi)
                left_y.append(targ[idx])

        r_x = np.array(right_x)
        r_y = np.array(right_y)
        l_x = np.array(left_x)
        l_y = np.array(left_y)
        return r_x, l_x, r_y, l_y
