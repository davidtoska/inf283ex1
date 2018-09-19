class SplitData:
    """Container object for split information"""

    def __init__(self, column_index):
        self.column_index = column_index
        self.is_numeric = False
        self.split_operator = None
        self.split_value = None
        self.information_gain = -1.0
        self.pure_value = None
        self.pure_value_count = 0

    def print(self):
        print("---- Split data object ----")
        print("column_index       : {}".format(self.column_index))
        print("is_numeric         : {}".format(self.is_numeric))
        print("split_value        : {}".format(self.split_value))
        print("information_gain   : {}".format(self.information_gain))

    def get_split_name(self):
        split_name = "feature" + str(self.column_index)
        if self.pure_value:
            split_name = str(self.pure_value) + "(" + str(self.pure_value_count) + ")"
        elif self.information_gain <= 0:
            split_name = "pure"
        elif self.is_numeric:
            split_name += " < " + str(self.split_value)
        else:
            split_name += "==" + self.split_value
        return split_name
