from sklearn.model_selection import train_test_split
from sklearn import svm, datasets

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


def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        clf = svm.SVC
    model = clf(**model_params)
    model.fit(x, y)
    return model


def read_digits():
    digits = datasets.load_digits()
    return digits.images, digits.target
