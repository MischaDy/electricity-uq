from typing import TYPE_CHECKING

from sklearn.linear_model import LinearRegression

from helpers import misc_helpers

if TYPE_CHECKING:
    import numpy as np


def train_linreg(
        X_train: 'np.ndarray',
        y_train: 'np.ndarray',
        X_val: 'np.ndarray',
        y_val: 'np.ndarray',
        n_jobs=-1,
) -> LinearRegression:
    X_train, y_train = misc_helpers.add_val_to_train(X_train, X_val, y_train, y_val)
    model = LinearRegression(n_jobs=n_jobs)
    model.fit(X_train, y_train)
    return model
