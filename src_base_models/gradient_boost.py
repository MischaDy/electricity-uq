from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from typing import TYPE_CHECKING, Literal

from helpers import misc_helpers

if TYPE_CHECKING:
    import numpy as np


def train_hgbr(
        X_train: 'np.ndarray',
        y_train: 'np.ndarray',
        X_val: 'np.ndarray',
        y_val: 'np.ndarray',
        cv_n_iter=100,
        cv_n_splits=10,
        model_param_distributions=None,
        categorical_features: list[int] = None,
        monotonic_cst: list[Literal[-1, 0, 1]] = None,
        random_seed=42,
        n_jobs=-1,
        verbose=1,
        val_frac=0.1,
        n_iter_no_change=30,
):
    X_train, y_train = misc_helpers.add_val_to_train(X_train, X_val, y_train, y_val)

    if model_param_distributions is None:
        model_param_distributions = {
            # 'max_features': stats.randint(1, X_train.shape[1]),
            "max_iter": stats.randint(10, 1000),
            'learning_rate': stats.loguniform(0.015, 0.15),
            'max_leaf_nodes': stats.randint(10, 100),
            'min_samples_leaf': stats.randint(15, 100),
            'l2_regularization': [0, 1e-4, 1e-3, 1e-2, 1e-1],
        }
    model = HistGradientBoostingRegressor(
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0,
        max_features=1.0,
        categorical_features=categorical_features,  # todo
        monotonic_cst=monotonic_cst,  # todo
        early_stopping=True,
        validation_fraction=val_frac,
        n_iter_no_change=n_iter_no_change,
        random_state=random_seed,
        verbose=0,
    )
    cv_obj = RandomizedSearchCV(
        model,
        param_distributions=model_param_distributions,
        n_iter=cv_n_iter,
        cv=TimeSeriesSplit(n_splits=cv_n_splits),
        scoring="neg_root_mean_squared_error",
        random_state=random_seed,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    cv_obj.fit(X_train, y_train.ravel())
    model = cv_obj.best_estimator_
    return model
