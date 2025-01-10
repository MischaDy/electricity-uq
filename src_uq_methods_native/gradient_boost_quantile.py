import logging
logging.basicConfig(level=logging.INFO)

logging.info('importing')  # temp

from concurrent.futures import ProcessPoolExecutor

from functools import partial

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from typing import Literal

from helpers import misc_helpers
import numpy as np


STORAGE_PATH = 'qhgbr_storage'


class HGBR_Quantile:
    def __init__(
            self,
            quantiles: list[float],
            max_iter=100,
            lr=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            max_features=1.0,
            categorical_features=None,
            monotonic_cst=None,
            val_frac=0.1,
            n_iter_no_change=30,
            random_seed=42,
    ):
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
            verbose=0,
        )
        self.models = {
            quantile: model_constructor(quantile=quantile)
            for quantile in self.quantiles
        }

    def fit(self, X_train, y_train, cv_n_iter=100, cv_n_splits=10, random_seed=42,
            model_param_distributions=None, verbose=0):
        from timeit import default_timer  # temp

        y_train = y_train.ravel()
        if model_param_distributions is None or cv_n_iter == 0:
            msg_param = 'cv_n_iter == 0' if cv_n_iter == 0 else 'model_param_distributions is None'
            msg = f'parameter {msg_param}, so no CV is performed'
            logging.info(msg)

            objs_dict = self.models
        else:
            logging.info('using CV')
            cv_maker = partial(
                RandomizedSearchCV,
                param_distributions=model_param_distributions,
                n_iter=cv_n_iter,
                cv=TimeSeriesSplit(n_splits=cv_n_splits),
                scoring="neg_root_mean_squared_error",
                random_state=random_seed,
                verbose=verbose,
                n_jobs=1,
            )
            objs_dict = {quantile: cv_maker(estimator=model) for quantile, model in self.models.items()}
        logging.info(f'fitting {len(objs_dict)} objects.')
        t1 = default_timer()
        # based on https://superfastpython.com/processpoolexecutor-map-vs-submit/
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.fit_obj, quantile=quantile, obj=obj, X_train=X_train, y_train=y_train,
                                       i=i)
                       for i, (quantile, obj) in enumerate(objs_dict.items(), start=1)]
        self.models = dict(future.result() for future in futures)
        t2 = default_timer()
        logging.info(f'finished after {t2 - t1}s')

    @staticmethod
    def fit_obj(quantile: float, obj: RandomizedSearchCV | HistGradientBoostingRegressor, X_train: np.ndarray,
                y_train: np.ndarray, i: int):  # type_: Literal['cv', 'model'] = 'cv'
        # prefix = f'{type_} {i} (q={quantile})'
        # logging.info(f'{prefix}: fitting...')
        obj.fit(X_train, y_train)
        # stop_iter = ... if type_ == 'cv' else ...
        # logging.info(f'{prefix}: done. best etimator stopped after {stop_iter} iterations.')
        return quantile, obj

    def predict(self, X_pred, as_dict=True):
        # todo: parallelize?
        result = {quantile: model.predict(X_pred)
                  for quantile, model in self.models.items()}
        if as_dict:
            return result
        return np.array(list(result.values()))


def train_hgbr_quantile(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        quantiles: list[float],
        cv_n_iter=100,
        cv_n_splits=10,
        model_param_distributions=None,
        categorical_features: list[int] = None,
        monotonic_cst: list[Literal[-1, 0, 1]] = None,
        random_seed=42,
        verbose=1,
        val_frac=0.1,
        n_iter_no_change=30,
        max_iter=1000,
        lr=0.2,
        l2_regularization=1e-3,
):
    if model_param_distributions is None:
        from scipy import stats
        model_param_distributions = {
            # 'max_features': stats.randint(1, X_train.shape[1]),
            # "max_iter": stats.randint(10, 1000),
            'learning_rate': [0.1, 0.15, 0.2],
            # 'max_leaf_nodes': stats.randint(10, 100),
            # 'min_samples_leaf': stats.randint(15, 100),
            'l2_regularization': [1e-4, 1e-3, 1e-2],
        }
    model = HGBR_Quantile(
        quantiles,
        categorical_features=categorical_features,
        monotonic_cst=monotonic_cst,
        val_frac=val_frac,
        n_iter_no_change=n_iter_no_change,
        random_seed=random_seed,
        max_iter=max_iter,
        lr=lr,
        l2_regularization=l2_regularization,
    )
    X_train, y_train = misc_helpers.add_val_to_train(X_train, X_val, y_train, y_val)
    model.fit(X_train, y_train, cv_n_iter=cv_n_iter, cv_n_splits=cv_n_splits, random_seed=random_seed,
              model_param_distributions=model_param_distributions, verbose=verbose)
    return model


def predict_with_hgbr_quantile(model: HGBR_Quantile, X_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_quantiles_dict = model.predict(X_pred, as_dict=True)
    y_pred = y_quantiles_dict[0.5]
    y_quantiles = misc_helpers.quantiles_dict_to_np_arr(y_quantiles_dict)
    y_std = misc_helpers.stds_from_quantiles(y_quantiles)
    return y_pred, y_quantiles, y_std


def plot_uq_worker(y_true_plot, y_pred_plot, ci_low_plot, ci_high_plot, label_part,
                   method, n_quantiles=None, show_plot=True, save_plot=True, plotting_90p_interval=False):
    from matplotlib import pyplot as plt
    base_title = method
    base_filename = method
    if plotting_90p_interval:
        label = '90% CI'
    else:
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
        plt.savefig(f'{filename}.png')
    if show_plot:
        plt.show(block=True)
    plt.close(fig)


def test_qhgbr():
    from scipy import stats
    import settings
    import settings_update

    from helpers.io_helper import IO_Helper

    logging.basicConfig(level=logging.INFO, force=True)

    io_helper = IO_Helper(STORAGE_PATH)

    logging.info('data setup...')

    SHOW_PLOT = False
    SAVE_PLOT = True
    PLOT_DATA = False
    RUN_SIZE = 'big'

    # if False, plot between outermost quantiles
    PLOT_90P_INTERVAL = True

    n_samples_plot = 1600

    val_frac = 0.1

    quantiles = settings.QUANTILES

    cv_n_iter = 0
    cv_n_splits = 2
    verbose = 2
    n_iter_no_change = 20
    max_iter = 1000
    lr = 0.2
    l2_regularization = 1e-3

    model_param_distributions = {
        # 'max_features': stats.randint(1, X_train.shape[1]),
        "max_iter": [1000],  # stats.randint(10, 1000),
        'learning_rate': [0.1],
        'l2_regularization': [1e-3],
    }

    ##############

    settings.RUN_SIZE = RUN_SIZE
    settings_update.update_run_size_setup()
    # settings.DATA_FILEPATH = f'../{settings.DATA_FILEPATH}'
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers.get_data(
        filepath=settings.DATA_FILEPATH,
        train_years=settings.TRAIN_YEARS,
        val_years=settings.VAL_YEARS,
        test_years=settings.TEST_YEARS,
        n_points_per_group=settings.N_POINTS_PER_GROUP,
        do_standardize_data=True,
    )

    if PLOT_DATA:
        logging.info('plotting data')
        from matplotlib import pyplot as plt
        plt.plot(y)
        plt.title('data')
        plt.show(block=True)

    X_pred = X
    y_true = y

    logging.info('training...')
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
        verbose=verbose,
        val_frac=val_frac,
        n_iter_no_change=n_iter_no_change,
        max_iter=max_iter,
        lr=lr,
        l2_regularization=l2_regularization,
    )
    prefix = 'qhgbr'
    postfix = f'n{X_train.shape[0]}_it{cv_n_iter}'

    io_helper.save_model(model, filename=f'{prefix}_y_pred_{postfix}.model')
    logging.info('predicting')
    y_pred, y_quantiles, y_std = predict_with_hgbr_quantile(model, X_pred)

    io_helper.save_array(y_pred, filename=f'{prefix}_y_pred_{postfix}.npy')
    io_helper.save_array(y_quantiles, filename=f'{prefix}_y_quantiles_{postfix}.npy')
    io_helper.save_array(y_std, filename=f'{prefix}_y_std_{postfix}.npy')

    y_pred, y_quantiles, y_std = y_pred[:n_samples_plot], y_quantiles[:n_samples_plot], y_std[:n_samples_plot]
    y_true = y_true[:n_samples_plot]

    index_low, index_high = [quantiles.index(0.05), quantiles.index(0.95)] if PLOT_90P_INTERVAL else [0, -1]
    ci_low, ci_high = y_quantiles[:, index_low], y_quantiles[:, index_high]
    n_quantiles = y_quantiles.shape[1]
    logging.info('plotting')
    plot_uq_worker(y_true, y_pred, ci_low, ci_high, 'full', 'qhgbr', n_quantiles, show_plot=SHOW_PLOT,
                   save_plot=SAVE_PLOT, plotting_90p_interval=PLOT_90P_INTERVAL)


if __name__ == '__main__':
    test_qhgbr()
