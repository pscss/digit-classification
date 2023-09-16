from sklearn.model_selection import train_test_split
from sklearn import svm, datasets, metrics
from sklearn.model_selection import ParameterGrid
import itertools

""" 
Common functions:
"""


# flatten the images
def preprocess_data(data):
    n = len(data)
    return data.reshape((n, -1))


def split_data(X, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def split_train_dev_test(X, y, test_size, dev_size, random_state=1):
    test_dev_size = test_size + dev_size
    if test_dev_size >= 0.9:
        raise ValueError(
            "Total test and Dev data cannot be more than 90% of entire data"
        )
    X_train, X_test_dev, y_train, y_test_dev = train_test_split(
        X,
        y,
        test_size=test_dev_size,
        random_state=random_state,
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_test_dev,
        y_test_dev,
        test_size=dev_size / test_dev_size,
        random_state=random_state,
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        clf = svm.SVC
    model = clf(**model_params)
    model.fit(x, y)
    return model


def read_digits():
    digits = datasets.load_digits()
    return digits.images, digits.target


def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    # cm = metrics.confusion_matrix(y_test, predicted)
    # print(f"Confusion matrix:\n{cm}")
    # print(
    #     f"Classification report for classifier {model}:\n"
    #     f"{metrics.classification_report(y_test, predicted)}\n"
    # )
    return metrics.accuracy_score(y_test, predicted)


def get_combinations_with_keys(grid):
    lists = grid.values()
    keys = grid.keys()
    combinations = list(itertools.product(*lists))
    return [dict(zip(keys, combination)) for combination in combinations]


def tune_hparams(X_train, X_dev, y_train, y_dev, h_params_grid):
    best_accuracy = -1
    best_model = None
    best_params = {}

    for h_params in ParameterGrid(h_params_grid):
        cur_model = train_model(X_train, y_train, h_params, model_type="svm")
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)

        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            best_params = h_params
            best_model = cur_model

    return best_model, best_params, best_accuracy
