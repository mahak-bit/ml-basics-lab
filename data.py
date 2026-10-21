"""
data.py - makes the fake (synthetic) datasets used by every other script.

Why fake data? Because we control it. We know which features actually matter,
so we can check whether our models figure that out.
"""

import os

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# All charts are saved in an "outputs" folder next to this file.
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "outputs")


def output_path(filename):
    """Return the full path for a file inside outputs/, creating the folder if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)


def get_classification_data():
    """
    A yes/no (binary) dataset where one answer is rare.
    About 90% of rows are class 0 and only 10% are class 1.
    This "imbalance" is on purpose - it makes accuracy misleading (see 03).
    """
    X, y = make_classification(
        n_samples=1000,  # 1000 rows
        n_features=10,  # 10 input columns
        n_informative=5,  # only 5 of them actually carry signal
        weights=[0.9, 0.1],  # 90% class 0, 10% class 1
        random_state=42,  # same "random" data every run
    )
    # Keep 80% of rows for training and hide 20% for testing.
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_regression_data():
    """
    A number-prediction dataset with 30 input columns.
    Only 5 columns are truly useful, so regularization has something to do.
    """
    X, y = make_regression(
        n_samples=200,
        n_features=30,
        n_informative=5,  # only 5 columns really matter
        noise=10,  # a little random wobble added to the answers
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


def main():
    # Print the shapes so we can see the data is what we expect.
    X_train, X_test, _, _ = get_classification_data()
    print("Classification data:")
    print(
        f"  train rows: {X_train.shape[0]}, test rows: {X_test.shape[0]}, features: {X_train.shape[1]}"
    )

    X_train, X_test, _, _ = get_regression_data()
    print("Regression data:")
    print(
        f"  train rows: {X_train.shape[0]}, test rows: {X_test.shape[0]}, features: {X_train.shape[1]}"
    )


if __name__ == "__main__":
    main()
