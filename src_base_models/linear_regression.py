from sklearn import linear_model
from typing import TYPE_CHECKING

from helpers import misc_helpers

if TYPE_CHECKING:
    import numpy as np


def train_linreg(
        X_train: 'np.ndarray',
        y_train: 'np.ndarray',
        X_val: 'np.ndarray',
        y_val: 'np.ndarray',
        n_jobs=-1,
):
    X_train, y_train = misc_helpers.add_val_to_train(X_train, X_val, y_train, y_val)
    model = linear_model.LinearRegression(n_jobs=n_jobs)
    model.fit(X_train, y_train)
    return model
