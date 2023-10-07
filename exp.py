"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

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

# Author: Prateek Singhal
import utils
from joblib import load
import pandas as pd
import argparse
import json


parser = argparse.ArgumentParser(
    description="Machine Learning Experimentation Script"
)
parser.add_argument(
    "--test-size", type=float, default=0.2, help="Size of the test set"
)
parser.add_argument(
    "--dev-size",
    type=float,
    default=0.2,
    help="Size of the development (dev) set",
)
parser.add_argument("--num-runs", type=int, default=5, help="Number of runs")
parser.add_argument(
    "--hyperparameters",
    type=str,
    required=True,
    help="JSON file containing hyperparameters",
)

# Parse command-line arguments
args = parser.parse_args()

# 1. Size parameters for setting up test
size_grid = {"test_size": [args.test_size], "dev_size": [args.dev_size]}
number_of_runs = args.num_runs

# 2. Model Hyperparameters
with open(args.hyperparameters, "r") as json_file:
    h_parameters_grid = json.load(json_file)

if not h_parameters_grid:
    raise ValueError("At-least one hyperparameter should be provided")

# 3. Data Sourcing
X, y = utils.read_digits()

print("Sample Information:")
print(f"Total Samples -> {X.shape[0]}")
print(f"Shape of the images in (height, width) -> {X.shape[1:]}")

results_list = []
combinations = utils.get_combinations_with_keys(size_grid)
for run_id in range(number_of_runs):
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
        for clf, h_params_grid in h_parameters_grid.items():
            best_model_path, best_params, dev_accuracy = utils.tune_hparams(
                X_train, X_dev, y_train, y_dev, h_params_grid, clf
            )
            best_model = load(best_model_path)
            train_accuracy = utils.predict_and_eval(
                best_model, X_train, y_train
            )
            test_accuracy = utils.predict_and_eval(best_model, X_test, y_test)
            print("Optimal parameters: ", best_params)
            print(
                f"model_type = {clf} test_size={test_size} dev_size={dev_size} train_size={train_size} train_acc={train_accuracy} dev_acc={dev_accuracy} test_acc={test_accuracy}"  # noqa
            )
            run_result_dict = {
                "run_id": run_id,
                "model_type": clf,
                "train_accuracy": round(train_accuracy, 2),
                "test_accuracy": round(test_accuracy, 2),
                "dev_accuracy": round(dev_accuracy, 2),
            }
            results_list.append(run_result_dict)

df = pd.DataFrame(results_list)
print(df.groupby("model_type").describe().T)
