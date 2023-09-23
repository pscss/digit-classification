"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""
# Author: Prateek Singhal


import itertools

import utils

# gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
# gamma_ranges = [0.001]
# C_ranges = [0.1, 1, 2, 5, 10]
# C_ranges = [2]
h_params_grid = {
    "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
    "C": [0.1, 1, 2, 5, 10],
}


###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

# 1. Data Sourcing
X, y = utils.read_digits()

print("Sample Information:")
print(f"Total Samples -> {X.shape[0]}")
print(f"Shape of the images in (height, width) -> {X.shape[1:]}")
print()

# QUIZ STATEMENTS


###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# 2. Data splitting -- to create train and test sets
# X_train, X_dev, X_test, y_train, y_dev, y_test = utils.split_train_dev_test(
#     X, y, test_size=0.2, dev_size=0.3
# )


# # 3. Data preprocessing
# X_train = utils.preprocess_data(X_train)
# X_dev = utils.preprocess_data(X_dev)
# X_test = utils.preprocess_data(X_test)


size_grid = {"test_size": [0.1, 0.2, 0.3], "dev_size": [0.1, 0.2, 0.3]}
combinations = utils.get_combinations_with_keys(size_grid)
for combination in combinations:
    test_size = combination["test_size"]
    dev_size = combination["dev_size"]
    train_size = 1 - (test_size + dev_size)
    if train_size <= 0:
        raise ValueError("Sum of test and dev should be less than 1")
    (
        X_train,
        X_dev,
        X_test,
        y_train,
        y_dev,
        y_test,
    ) = utils.split_train_dev_test(X, y, test_size, dev_size)
    X_train = utils.preprocess_data(X_train)
    X_dev = utils.preprocess_data(X_dev)
    X_test = utils.preprocess_data(X_test)
    best_model, best_params, dev_accuracy = utils.tune_hparams(
        X_train, X_dev, y_train, y_dev, h_params_grid
    )
    train_accuracy = utils.predict_and_eval(best_model, X_train, y_train)
    test_accuracy = utils.predict_and_eval(best_model, X_test, y_test)
    # print("Optimal parameters: ", best_params)
    print(
        f"test_size={test_size} dev_size={dev_size} train_size={train_size} train_acc={train_accuracy} dev_acc={dev_accuracy} test_acc={test_accuracy}"  # noqa
    )


# HYPER PARAMETERS TUNING
# best_model, best_params, best_accuracy = utils.tune_hparams(
#     X_train, X_dev, y_train, y_dev, h_params_grid
# )
# print("Optimal parameters: ", best_params)

# 4. Model initialization and model fit
# Create a classifier: a support vector classifier
# model = utils.train_model(X_train, y_train, {"gamma": 0.001, "C": 2})


# 5. Predict and Evaluation:
# accuracy = utils.predict_and_eval(best_model, X_dev, y_dev)
# if accuracy < 0.8:
#     raise ValueError("Change parameters to get higher accuracy than 80%.")


# 6. Testing on Test data:
# print()
# print("Testing model for Robustness")
# test_accuracy = utils.predict_and_eval(best_model, X_dev, y_dev)
# print(
#     f"""
# Validation and Test accuracies of the model are:
# validation accuracy {accuracy}
# test accuracy   {test_accuracy}
# """
# )
