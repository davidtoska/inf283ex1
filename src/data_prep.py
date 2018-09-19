import numpy as np


def get_iris_data():
    """
    Utility function that loads the iris dataset, and returns the
    feature values (X), target values (y), feature names and target name

    :return:
    """

    target_name = "species"
    feature_names = np.array(["sepal_length",
                              "sepal_width",
                              "petal_length",
                              "petal_width"])
    X = []
    y = []

    # Load iris-data
    with open("../data/iris_data.csv") as iris_csv:
        for line in iris_csv:
            # Read the comma separated values
            x1, x2, x3, x4, y1_raw = line.split(",")
            y1 = y1_raw.rstrip('`\n')
            # Append element to lists
            x1 = float(x1)
            x2 = float(x2)
            x3 = float(x3)
            x4 = float(x4)
            X.append([x1, x2, x3, x4])
            y.append(y1)

    X = np.array(X)
    y = np.array(y)

    return X, y, feature_names, target_name


def get_mushroom_data():
    """Utility function that returns X, y, feature_names, and target_name. """

    target_name = ""
    feature_names = np.array(["cap-shape",
                              "cap-surface",
                              "cap-color",
                              "bruises",
                              "odor",
                              "gill-attachment",
                              "gill-spacing",
                              "gill-size",
                              "gill-color",
                              "stalk-shape",
                              "stalk-root",
                              "stalk-surface-above-ring",
                              "stalk-surface-below-ring",
                              "stalk-color-above-ring",
                              "stalk-color-below-ring",
                              "veil-type",
                              "veil-color",
                              "ring-number",
                              "ring-type",
                              "spore-print-color",
                              "population",
                              "habitat"])

    X = []
    y = []

    # Load mushroom-data
    for line in open("../data/mushroom_data.csv"):
        row = line.split(",")
        row[-1] = row[-1].rstrip('\n')
        x_i = row[1:]
        y_i = row[0]
        X.append(x_i)
        y.append(y_i)

    X = np.array(X)
    y = np.array(y)

    return X, y, feature_names, target_name
