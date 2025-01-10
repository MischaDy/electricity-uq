from functools import partial

from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from typing import TYPE_CHECKING, Literal

from helpers import misc_helpers

if TYPE_CHECKING:
    import numpy as np


class HGBR_Quantile:
    def __init__(
            self,
            quantiles: list[float],
            max_iter=100,
            lr=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0,
            max_features=1.0,
            categorical_features=None,
            monotonic_cst=None,
            val_frac=0.1,
            n_iter_no_change=30,
            random_seed=42,
            verbose=0,
    ):
        self.verbose = verbose
        self.quantiles = sorted(quantiles)
        model_constructor = partial(
            HistGradientBoostingRegressor,
            loss='quantile',
            max_iter=max_iter,
            learning_rate=lr,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_features=max_features,
            categorical_features=categorical_features,  # todo
            monotonic_cst=monotonic_cst,  # todo
            early_stopping=True,
            validation_fraction=val_frac,
            n_iter_no_change=n_iter_no_change,
            random_state=random_seed,
            verbose=verbose,
        )
        self.models = {
            quantile: model_constructor(quantile)
            for quantile in self.quantiles
        }

    def fit(self, X_train, y_train, cv_n_iter=100, cv_n_splits=10, n_jobs=-1, random_seed=42,
            model_param_distributions=None):
        if model_param_distributions is None:
            model_param_distributions = {
                # 'max_features': stats.randint(1, X_train.shape[1]),
                "max_iter": stats.randint(10, 1000),
                'learning_rate': stats.loguniform(0.015, 0.15),
                'max_leaf_nodes': stats.randint(10, 100),
                'min_samples_leaf': stats.randint(15, 100),
                'l2_regularization': [0, 1e-4, 1e-3, 1e-2, 1e-1],
            }
        cv_maker = partial(
            RandomizedSearchCV,
            param_distributions=model_param_distributions,
            n_iter=cv_n_iter,
            cv=TimeSeriesSplit(n_splits=cv_n_splits),
            scoring="neg_root_mean_squared_error",
            random_state=random_seed,
            verbose=self.verbose,
            n_jobs=n_jobs,
        )
        cv_objs = {quantile: cv_maker(model) for quantile, model in self.models.items()}
        y_train = y_train.ravel()

        # todo: parallelize!
        for cv_obj in cv_objs.values():
            cv_obj.fit(X_train, y_train)
        self.models = {quantile: cv_obj.best_estimator_ for quantile, cv_obj in cv_objs.items()}

    def predict(self, X_pred, as_dict=True):
        # todo: parallelize?
        result = {quantile: model.predict(X_pred)
                  for quantile, model in self.models.items()}
        if as_dict:
            return result
        return np.array(list(result.values()))


def train_hgbr_quantile(
        X_train: 'np.ndarray',
        y_train: 'np.ndarray',
        X_val: 'np.ndarray',
        y_val: 'np.ndarray',
        quantiles: list[float],
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
    if model_param_distributions is None:
        model_param_distributions = {
            # 'max_features': stats.randint(1, X_train.shape[1]),
            "max_iter": stats.randint(10, 1000),
            'learning_rate': stats.loguniform(0.015, 0.15),
            'max_leaf_nodes': stats.randint(10, 100),
            'min_samples_leaf': stats.randint(15, 100),
            'l2_regularization': [0, 1e-4, 1e-3, 1e-2, 1e-1],
        }
    model = HGBR_Quantile(
        quantiles,
        categorical_features=categorical_features,
        monotonic_cst=monotonic_cst,
        val_frac=val_frac,
        n_iter_no_change=n_iter_no_change,
        random_seed=random_seed,
        verbose=verbose,
    )
    X_train, y_train = misc_helpers.add_val_to_train(X_train, X_val, y_train, y_val)
    model.fit(X_train, y_train, cv_n_iter=cv_n_iter, cv_n_splits=cv_n_splits, n_jobs=n_jobs, random_seed=random_seed,
              model_param_distributions=model_param_distributions)
    return model


def predict_with_hgbr_quantile(model: HGBR_Quantile, X_pred: np.array):
    y_quantiles = model.predict(X_pred, as_dict=True)
    y_pred = y_quantiles[0.5]
    y_std = misc_helpers.stds_from_quantiles(y_quantiles)
    return y_pred, y_quantiles, y_std
