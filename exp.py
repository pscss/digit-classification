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
gamma_ranges = [0.001]
# C_ranges = [0.1, 1, 2, 5, 10]
C_ranges = [2]


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
X_train, X_dev, X_test, y_train, y_dev, y_test = utils.split_train_dev_test(
    X, y, test_size=0.2, dev_size=0.3
)


# 3. Data preprocessing
X_train = utils.preprocess_data(X_train)
X_dev = utils.preprocess_data(X_dev)
X_test = utils.preprocess_data(X_test)


# HYPER PARAMETERS TUNING

# -take all combinations of gamma and C_ranges
best_accuracy_so_far = -1
best_model = None

for cur_gamma, cur_C in itertools.product(gamma_ranges, C_ranges):
    cur_model = utils.train_model(
        X_train, y_train, {"gamma": cur_gamma, "C": cur_C}, model_type="svm"
    )
    cur_accuracy = utils.predict_and_eval(cur_model, X_dev, y_dev)
    if cur_accuracy > best_accuracy_so_far:
        print("New best accuracy: ", cur_accuracy)
        best_accuracy_so_far = cur_accuracy
        optimal_gamma = cur_gamma
        optimal_C = cur_C
        best_model = cur_model
print("Optimal parameters gamma: ", optimal_gamma, "C: ", optimal_C)

# 4. Model initialization and model fit
# Create a classifier: a support vector classifier
# model = utils.train_model(X_train, y_train, {"gamma": 0.001})


# 5. Predict and Evaluation:
accuracy = utils.predict_and_eval(best_model, X_dev, y_dev)
# if accuracy < 0.8:
#     raise ValueError("Change parameters to get higher accuracy than 80%.")


# 6. Testing on Test data:
# print()
# print("Testing model for Robustness")
test_accuracy = utils.predict_and_eval(best_model, X_dev, y_dev)
print(
    f"""
Validation and Test accuracies of the model are:
validation accuracy {accuracy}
test accuracy   {test_accuracy}
"""
)


###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")


# cm = metrics.confusion_matrix(y_test, predicted)
# print(f"Confusion matrix:\n{cm}")

# plt.show()

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
# y_true = []
# y_pred = []
# # cm = disp.confusion_matrix

# # For each cell in the confusion matrix, add the corresponding ground truths
# # and predictions to the lists
# for gt in range(len(cm)):
#     for pred in range(len(cm)):
#         y_true += [gt] * cm[gt][pred]
#         y_pred += [pred] * cm[gt][pred]

# print(
#     "Classification report rebuilt from confusion matrix:\n"
#     f"{metrics.classification_report(y_true, y_pred)}\n"
# )
