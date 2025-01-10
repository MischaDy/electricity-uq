import logging
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


def plot_uq_worker(y_true_plot, y_pred_plot, ci_low_plot, ci_high_plot, label_part,
                   method, n_quantiles=None, show_plot=True, save_plot=True):
    from matplotlib import pyplot as plt
    base_title = method
    base_filename = method
    label = f'outermost 2/{n_quantiles} quantiles' if n_quantiles is not None else 'outermost 2 quantiles'
    x_plot = np.arange(y_true_plot.shape[0])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    ax.plot(x_plot, y_true_plot, label=f'{label_part} data', color="black", linestyle='dashed')
    ax.plot(x_plot, y_pred_plot, label="point prediction", color="green")
    ax.fill_between(
        x_plot,
        ci_low_plot,
        ci_high_plot,
        color="green",
        alpha=0.2,
        label=label,
    )
    ax.legend()
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title(f'{base_title} ({label_part})')
    if save_plot:
        filename = f'{base_filename}_{label_part}'
        plt.savefig(f'../comparison_storage/plots/{filename}.png')
    if show_plot:
        plt.show(block=True)
    plt.close(fig)


def test_qhgbr():
    import settings

    logging.basicConfig(level=logging.INFO)
    logging.info('data setup...')

    SHOW_PLOT = True
    SAVE_PLOT = True
    PLOT_DATA = True
    USE_REAL_DATA = True

    n_samples = 1600

    train_frac = 0.4
    val_frac = 0.1
    test_frac = 0.5

    quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]  # settings.QUANTILES

    cv_n_iter = 10
    cv_n_splits = 10
    verbose = 1
    n_iter_no_change = 30

    model_param_distributions = {
        # 'max_features': stats.randint(1, X_train.shape[1]),
        "max_iter": stats.randint(10, 1000),
        'learning_rate': stats.loguniform(0.015, 0.15),
        'max_leaf_nodes': stats.randint(10, 100),
        'min_samples_leaf': stats.randint(15, 100),
        'l2_regularization': [0, 1e-4, 1e-3, 1e-2, 1e-1],
    }

    ##############

    n_train_samples = round(train_frac * n_samples)
    n_val_samples = round(val_frac * n_samples)
    n_test_samples = round(test_frac * n_samples)

    # if USE_REAL_DATA:
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers.get_data(
        '../data/data_1600.pkl',
        n_points_per_group=n_samples,
    )
    # else:
    #     dim = 10
    #     X = np.arange(n_samples * dim).reshape(n_samples, dim)
    #     y = np.sin(X.sum(axis=1) / n_samples / 3).reshape(-1, 1)

    if PLOT_DATA:
        logging.info('plotting data')
        from matplotlib import pyplot as plt
        plt.plot(y)
        plt.show(block=True)

    X_train = X[:n_train_samples]
    y_train = y[:n_train_samples]
    X_val = X[n_train_samples:n_train_samples + n_val_samples]
    y_val = y[n_train_samples:n_train_samples + n_val_samples]
    X_test = X[-n_test_samples:]
    y_test = y[-n_test_samples:]

    X_pred = X
    y_true = y

    model = train_hgbr_quantile(
        X_train,
        y_train,
        X_val,
        y_val,
        quantiles,
        cv_n_iter=cv_n_iter,
        cv_n_splits=cv_n_splits,
        model_param_distributions=model_param_distributions,
        random_seed=42,
        n_jobs=-1,
        verbose=verbose,
        val_frac=val_frac,
        n_iter_no_change=n_iter_no_change,
    )
    y_pred, y_quantiles, y_std = predict_with_hgbr_quantile(model, X_pred)
    ci_low, ci_high = (
        y_quantiles[:, 0],
        y_quantiles[:, -1],
    )
    n_quantiles = y_quantiles.shape[1]
    plot_uq_worker(y_true, y_pred, ci_low, ci_high, 'full', 'qr', n_quantiles, show_plot=SHOW_PLOT, save_plot=SAVE_PLOT)


if __name__ == '__main__':
    test_qhgbr()
