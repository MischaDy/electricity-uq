from abc import ABC, abstractmethod
from itertools import chain
from typing import Iterable

import numpy as np
from mapie.subsample import BlockBootstrap
from matplotlib import pyplot as plt

from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from uncertainty_toolbox import get_all_metrics

from helpers import get_data, starfilter
from mapie_plot_ts_tutorial import (
    train_base_model,
    estimate_pred_interals_no_pfit_enbpi,
)
from quantile_regression import estimate_quantiles as estimate_quantiles_qr


PLOT_DATA = False


# noinspection PyPep8Naming
class UQ_Comparer(ABC):
    """
    Usage:
    1. inherit from this class
    2. override get_data, compute_metrics, and train_base_model
    3. define all desired posthoc and native UQ methods. all posthoc and native UQ method names should start with
       'posthoc_' and 'native_', respectively
    """

    # todo: add type hints
    # todo: make classmethod?
    @abstractmethod
    def get_data(self) -> Iterable[...]:
        raise NotImplementedError

    # todo: add type hints
    # todo: make classmethod?
    @abstractmethod
    def compute_metrics(self, y_pred: ..., y_pis: ..., y_true: ...) -> ...:
        raise NotImplementedError

    # todo: add type hints
    # todo: make classmethod?
    @abstractmethod
    def train_base_model(self, X_train: ..., y_train: ...) -> ...:
        raise NotImplementedError

    @classmethod
    def get_posthoc_methods(cls):
        return cls._get_methods_by_prefix("posthoc")

    @classmethod
    def get_native_methods(cls):
        return cls._get_methods_by_prefix("native")

    @classmethod
    def _get_methods_by_prefix(cls, prefix):
        return dict(
            starfilter(
                lambda k, v: k.startswith(prefix), cls.__dict__.items()
            )
        )

    # todo: make classmethod?
    def run_posthoc_methods(self, X_train, y_train, X_test):
        return self._run_methods(X_train, y_train, X_test, uq_type="posthoc")

    # todo: make classmethod?
    def run_native_methods(self, X_train, y_train, X_test):
        return self._run_methods(X_train, y_train, X_test, uq_type="native")

    # todo: make classmethod?
    def _run_methods(self, X_train, y_train, X_test, *, uq_type):
        """

        :param X_train:
        :param y_train:
        :param X_test:
        :param uq_type: one of: "posthoc", "native"
        :return:
        """
        assert uq_type in ["posthoc", "native"]
        is_posthoc = uq_type == "posthoc"
        if is_posthoc:
            print("training base model")
            base_model = self.train_base_model(X_train, y_train)
        print(f"running {uq_type} methods")
        uq_methods: Iterable[tuple] = starfilter(
            lambda k, v: k.startswith(uq_type), self.__class__.__dict__.items()
        )
        uq_results = {}
        for method_name, method in uq_methods:
            # todo: pass deepcopy instead?
            # noinspection PyUnboundLocalVariable
            y_pred, y_pis = (
                method(base_model, X_train, y_train, X_test)
                if is_posthoc
                else method(X_train, y_train, X_test)
            )
            uq_results[method_name] = y_pred, y_pis
        return uq_results


# noinspection PyPep8Naming
class My_UQ_Comparer(UQ_Comparer):
    def get_data(self):
        N_POINTS_TEMP = 100  # per group
        return get_data(N_POINTS_TEMP, return_full_data=True)

    def compute_metrics(self, y_pred, y_pis, y_true):
        return get_all_metrics(
            # todo: correct?
            y_pred,
            y_pis[:, 1] - y_pis[:, 0],
            y_true.to_numpy().squeeze(),
        )

    def train_base_model(self, X_train, y_train, model_params_choices=None):
        if model_params_choices is None:
            model_params_choices = {
                "max_depth": randint(2, 30),
                "n_estimators": randint(10, 100),
            }
        return train_base_model(
            RandomForestRegressor,
            model_params_choices=model_params_choices,
            X_train=X_train,
            y_train=y_train,
            load_trained_model=True,
            cv_n_iter=10,
        )

    @staticmethod
    def posthoc_conformal_prediction(model, X_train, y_train, X_test):
        random_state = 59
        alpha = 0.05
        cv_mapie_ts = BlockBootstrap(
            n_resamplings=10, n_blocks=10, overlapping=False, random_state=random_state
        )
        y_pred, y_pis = estimate_pred_interals_no_pfit_enbpi(
            model,
            cv_mapie_ts,
            alpha,
            X_test,
            X_train,
            y_train,
            skip_base_training=True,
        )

        return y_pred, y_pis

    @staticmethod
    def native_quantile_regression(X_train, y_train, X_test, ci_alpha=0.1):
        return estimate_quantiles_qr(X_train, y_train, X_test, ci_alpha=ci_alpha)


def compare_methods(
    uq_comparer: UQ_Comparer,
    should_plot_data=True,
) -> tuple[dict[str, tuple[np.array, np.array]], dict[str, tuple[np.array, np.array]]]:
    """

    :param uq_comparer:
    :param should_plot_data:
    :return: native_results, posthoc_results
    """
    print("loading data")
    X_train, X_test, y_train, y_test, X, y = uq_comparer.get_data()
    print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    if should_plot_data:
        print("plotting data")
        plot_data(X_train, X_test, y_train, y_test)

    print("running native methods")
    native_results = uq_comparer.run_native_methods(X_train, y_train, X_test)
    posthoc_results = uq_comparer.run_posthoc_methods(X_train, y_train, X_test)

    print("plotting native vs posthoc results")
    plot_intervals(X_train, y_train, X_test, y, native_results, posthoc_results)

    # todo
    # print("computing metrics")
    # native_metrics, posthoc_metrics = UQ_Comparer.compute_metrics(
    #     metrics, native_results, posthoc_results, y_test
    # )
    return native_results, posthoc_results


def compute_metrics(metrics, native_results, posthoc_results, y_test):
    native_metrics, posthoc_metrics = {}, {}
    for method_name, (y_pred, y_pis) in chain(
        native_results.items(), posthoc_results.items()
    ):
        print(f"\n========= {method_name} =========\n")
        metric_dict = (
            native_metrics if method_name in native_results else posthoc_metrics
        )
        for metric_name, metric_func in metrics.items():
            metric_value = metric_func(y_pred, y_pis, y_test)
            metric_dict[metric_name] = metric_value
            print(f"{metric_name}: {metric_value}")

    # acc_cp, acc_qr = ..., ...  # todo
    # print('accuracy CP vs QR:', acc_cp, acc_qr)
    #
    # crps_cp, crps_qr = ..., ...  # todo
    # print('CRPS CP vs QR:', crps_cp, crps_qr)
    return native_metrics, posthoc_metrics


# PLOTTING


def plot_data(
    X_train,
    X_test,
    y_train,
    y_test,
    figsize=(16, 5),
    ylabel="energy data",  # todo: details!
):
    """visualize training and test sets"""
    num_train_steps = X_train.shape[0]
    num_test_steps = X_test.shape[0]

    x_plot_train = np.arange(num_train_steps)
    x_plot_test = x_plot_train + num_test_steps

    plt.figure(figsize=figsize)
    plt.plot(x_plot_train, y_train)
    plt.plot(x_plot_test, y_test)
    plt.ylabel(ylabel)
    plt.legend(["Training data", "Test data"])
    plt.show()


def plot_intervals(X_train, y_train, X_test, y, native_results, posthoc_results):
    res_dict = {"native": native_results, "posthoc": posthoc_results}
    for res_type, results in res_dict.items():
        print(f"plotting {res_type} results...")
        for method_name, (y_preds, y_pis) in results.items():
            uq_type, *method_name_parts = method_name.split('_')
            plot_uq_results(X_train, X_test, y_train, y, y_pis, y_preds, plot_name=' '.join(method_name_parts), uq_type=uq_type)


def plot_uq_results(X_train, X_test, y_train, y, y_pis, y_preds, plot_name, uq_type):
    num_train_steps = X_train.shape[0]
    num_test_steps = X_test.shape[0]
    num_steps_total = num_train_steps + num_test_steps

    x_plot_train = np.arange(num_train_steps)
    x_plot_test = x_plot_train + num_test_steps
    x_plot_full = np.arange(num_steps_total)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    ax.plot(x_plot_full, y, color="black", linestyle="dashed", label="True mean")
    ax.scatter(
        x_plot_train,
        y_train,
        color="black",
        marker="o",
        alpha=0.8,
        label="training points",
    )
    ax.plot(
        x_plot_test,
        y_preds,
        label=f"mean/median prediction {plot_name}",  # todo: mean or median?
        color="green",
    )
    ax.fill_between(
        x_plot_test.ravel(),
        y_pis[:, 0],
        y_pis[:, 1],
        color="green",
        alpha=0.2,
        label=rf"{plot_name} 95% CI",  # todo: should depend on alpha!
    )
    ax.legend()
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title(f'{plot_name} ({uq_type})')
    plt.show()


if __name__ == "__main__":
    compare_methods(
        My_UQ_Comparer(),
        should_plot_data=PLOT_DATA,
    )
