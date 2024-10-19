import copy
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from helpers import starfilter


# todo: add type hints
# noinspection PyPep8Naming
class UQ_Comparer(ABC):
    # todo: After these required inputs, they may accept any args or kwargs you wish.
    """
    Usage:
    1. Inherit from this class.
    2. Override get_data, compute_metrics, and train_base_model.
    3. Define all desired posthoc and native UQ methods. The required signature is:
            (X_train, y_train, X_test, quantiles) -> (y_pred, y_quantiles, y_std)
       Posthoc methods receive an additional base_model parameter, so their signature looks like:
            (..., quantiles, base_model, *args, **kwargs) -> (y_pred, ...)
       All posthoc and native UQ method names should start with 'posthoc_' and 'native_', respectively. They should
       all be instance methods, not class or static methods.
    4. Call compare_methods from the child class
    """

    def compare_methods(
        self,
        quantiles,
        should_plot_data=True,
        should_plot_results=True,
        should_save_plots=True,
        plots_path=".",
        return_results=False,
        skip_deepcopy=False,
    ):  # -> tuple[dict[str, tuple[np.array, np.array]], dict[str, tuple[np.array, np.array]]]
        """
        :param skip_deepcopy:
        :param plots_path:
        :param should_save_plots:
        :param quantiles:
        :param should_plot_data:
        :param should_plot_results:
        :param return_results: return native and posthoc results in addition to the native and posthoc metrics?
        :return: native and posthoc metrics if return_results is False, else also native and posthoc results
        """
        print("loading data")
        X_train, X_test, y_train, y_test, X, y = self.get_data()
        print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        if should_plot_data:
            print("plotting data")
            plot_data(
                X_train,
                X_test,
                y_train,
                y_test,
                save_plot=should_save_plots,
                plots_path=plots_path,
            )

        print("running UQ methods")
        native_results = self.run_native_methods(
            X_train, y_train, X_test, quantiles=quantiles
        )
        posthoc_results = self.run_posthoc_methods(
            X_train, y_train, X_test, quantiles=quantiles, skip_deepcopy=skip_deepcopy
        )

        if should_plot_results:
            print("plotting native vs posthoc results")
            plot_intervals(
                X_train,
                y_train,
                X_test,
                y,
                native_results,
                posthoc_results,
                save_plots=should_save_plots,
                plots_path=plots_path,
            )

        print("computing and comparing metrics")
        native_metrics, posthoc_metrics = (
            self.compute_all_metrics(results, y_test, quantiles=quantiles)
            for results in [native_results, posthoc_results]
        )
        if return_results:
            return native_metrics, posthoc_metrics, native_results, posthoc_results
        return native_metrics, posthoc_metrics

    # todo: make classmethod?
    @abstractmethod
    def get_data(
        self,
    ) -> tuple[
        npt.NDArray[float],
        npt.NDArray[float],
        npt.NDArray[float],
        npt.NDArray[float],
        npt.NDArray[float],
        npt.NDArray[float],
    ]:
        """

        :return: X_train, X_test, y_train, y_test, X, y
        """
        raise NotImplementedError

    # todo: make classmethod?
    @abstractmethod
    def compute_metrics(
        self,
        y_pred,
        y_quantiles: Optional[npt.NDArray],
        y_std: Optional[npt.NDArray],
        y_true,
        quantiles,
    ):
        """
        Evaluate a UQ method by compute all desired metrics.

        :param y_pred:
        :param y_quantiles: array of shape (n_samples, n_quantiles) containing the predicted quantiles of y_true
        :param y_std: 1D array-like of predicted standard deviations around y_true
        :param y_true:
        :param quantiles: list of quantiles with which test to measure performance. They are expected to be symmetric
        :return:
        """
        raise NotImplementedError

    # todo: make classmethod?
    def compute_all_metrics(
        self,
        uq_results: dict[
            str,
            tuple[
                npt.NDArray[float],
                Optional[npt.NDArray[float]],
                Optional[npt.NDArray[float]],
            ],
        ],
        y_true,
        quantiles=None,
    ):
        """

        :param quantiles: quantiles
        :param uq_results: dict of (method_name, (y_pred, y_quantiles, y_std))
        :param y_true:
        :return:
        """
        return {
            method_name: self.compute_metrics(
                y_pred, y_quantiles, y_std, y_true, quantiles=quantiles
            )
            for method_name, (y_pred, y_quantiles, y_std) in uq_results.items()
        }

    @abstractmethod
    def train_base_model(self, X_train, y_train):
        raise NotImplementedError

    @classmethod
    def get_posthoc_methods(cls):
        return cls._get_uq_methods_by_type("posthoc")

    @classmethod
    def get_native_methods(cls):
        return cls._get_uq_methods_by_type("native")

    def _get_uq_methods_by_type(self, uq_type: str):
        """

        :param uq_type: one of "native", "posthoc"
        :return: all instance methods (i.e. callable attributes) with prefix given by uq_type
        """
        for attr_name in self.__class__.__dict__.keys():
            attr = getattr(self, attr_name)
            if attr_name.startswith(uq_type) and callable(attr):
                yield attr_name, attr

    def run_posthoc_methods(
        self, X_train, y_train, X_test, quantiles, skip_deepcopy=False,
    ):
        return self._run_methods(
            X_train, y_train, X_test, quantiles=quantiles, uq_type="posthoc", skip_deepcopy=skip_deepcopy,
        )

    def run_native_methods(self, X_train, y_train, X_test, quantiles):
        return self._run_methods(
            X_train, y_train, X_test, quantiles=quantiles, uq_type="native"
        )

    def _run_methods(
        self, X_train, y_train, X_test, quantiles, *, uq_type, skip_deepcopy=False,
    ) -> dict[str, tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]]:
        """

        :param skip_deepcopy: whether to skip making a deepcopy of the base model. speed up execution, but can lead to bugs if posthoc method affects the base model object. ignored for native methods
        :param X_train:
        :param y_train:
        :param X_test:
        :param uq_type: one of: "posthoc", "native"
        :return: dict of (method_name, (y_pred, y_quantiles)), where y_pred and y_quantiles are 1D and 2D, respectively
        """
        assert uq_type in ["posthoc", "native"]
        is_posthoc = uq_type == "posthoc"
        if is_posthoc:
            print("training base model")
            base_model = self.train_base_model(X_train, y_train)
        print(f"running {uq_type} methods")
        uq_results = {}
        for method_name, method in self._get_uq_methods_by_type(uq_type):
            if is_posthoc:
                # noinspection PyUnboundLocalVariable
                base_model_copy = (
                    copy.deepcopy(base_model) if not skip_deepcopy else base_model
                )
                y_pred, y_quantiles, y_std = method(
                    X_train, y_train, X_test, quantiles, base_model_copy
                )
            else:
                y_pred, y_quantiles, y_std = method(X_train, y_train, X_test, quantiles)
            uq_results[method_name] = y_pred, y_quantiles, y_std
        return uq_results

    @staticmethod
    def stds_from_quantiles(quantiles: npt.NDArray):
        """
        :param quantiles: array of shape (number of datapoints, number of quantiles), where number of quantiles should
        be at least about 100
        :return:
        """
        num_quantiles = quantiles.shape[1]
        if num_quantiles < 50:
            print(
                f"warning: {num_quantiles} quantiles are too few to compute a reliable std from (should be about 100)"
            )
        return np.std(quantiles, ddof=1, axis=1)

    @staticmethod
    def pis_from_quantiles(quantiles):
        mid = len(quantiles) // 2
        first, second = quantiles[:mid], quantiles[mid:]
        pi_limits = zip(first, reversed(second))
        pis = [high - low for low, high in pi_limits]
        return pis

    @staticmethod
    def optional(func):
        # todo
        """
        :param func:
        :return:
        """

        def optional_func(*args, **kwargs):
            pass


# PLOTTING


def plot_data(
    X_train,
    X_test,
    y_train,
    y_test,
    figsize=(16, 5),
    ylabel="energy data",  # todo: details!
    save_plot=True,
    filename="data.png",
    plots_path=".",
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
    if save_plot:
        filepath = os.path.join(plots_path, filename)
        plt.savefig(filepath)
    plt.show()


def plot_intervals(
    X_train,
    y_train,
    X_test,
    y,
    native_results,
    posthoc_results,
    save_plots=True,
    plots_path=".",
):
    res_dict = {"native": native_results, "posthoc": posthoc_results}
    for res_type, results in res_dict.items():
        print(f"plotting {res_type} results...")
        # todo: allow results to have multiple PIs (corresp. to multiple alphas)?
        for method_name, (y_preds, y_quantiles, y_std) in results.items():
            uq_type, *method_name_parts = method_name.split("_")
            plot_uq_results(
                X_train,
                X_test,
                y_train,
                y,
                y_preds,
                y_quantiles,
                y_std,
                plot_name=" ".join(method_name_parts),
                uq_type=uq_type,
                save_plot=save_plots,
                plots_path=plots_path,
            )


def plot_uq_results(
    X_train,
    X_test,
    y_train,
    y,
    y_preds,
    y_quantiles,
    y_std,
    plot_name,
    uq_type,
    save_plot=True,
    plots_path=".",
):
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

    if y_std is None:
        ci_low, ci_high = (
            y_quantiles[:, 0],
            y_quantiles[:, -1],
        )
    else:
        ci_low, ci_high = y_preds - y_std / 2, y_preds + y_std / 2
    ax.fill_between(
        x_plot_test.ravel(),
        ci_low,
        ci_high,
        color="green",
        alpha=0.2,
        label=rf"{plot_name} CI",  # todo: should depend on alpha!
    )
    ax.legend()
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title(f"{plot_name} ({uq_type})")
    if save_plot:
        filename = f"{plot_name}_{uq_type}.png"
        filepath = os.path.join(plots_path, filename)
        plt.savefig(filepath)
    plt.show()
