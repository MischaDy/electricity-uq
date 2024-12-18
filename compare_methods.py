import os
filename = os.path.split(__file__)[-1]
print(f'reading file {filename}...')

from abc import ABC, abstractmethod
import numpy as np
import copy
from collections import defaultdict
from functools import partial
from typing import Any, Generator, Callable

from io_helper import IO_Helper
from helpers import starfilter


DataTuple = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


# todo: add type hints
# noinspection PyPep8Naming
class UQ_Comparer(ABC):
    # todo: After these required inputs, they may accept any args or kwargs you wish.
    # todo: update usage guide!
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

    def __init__(self, storage_path, method_whitelist=None, posthoc_base_blacklist: dict[str, str] | None = None,
                 standardize_data=True,):
        """

        :param storage_path:
        :param method_whitelist:
        :param posthoc_base_blacklist:
        :param standardize_data: True if both X and y should be standardized, False if neither.
        """
        self.io_helper = IO_Helper(storage_path)
        # todo: store train and test data once loaded
        self.methods_kwargs = defaultdict(dict)
        self.method_whitelist = method_whitelist
        if posthoc_base_blacklist is None:
            posthoc_base_blacklist = dict()
        self.posthoc_base_blacklist = defaultdict(set)
        self.posthoc_base_blacklist.update(posthoc_base_blacklist)
        self.standardize_data = standardize_data

    def compare_methods(
            self,
            quantiles,
            should_plot_data=True,
            should_plot_uq_results=True,
            should_plot_base_results=True,
            should_show_plots=True,
            should_save_plots=True,
            skip_base_model_copy=False,
    ) -> tuple[dict, dict] | dict[str, dict[str, dict[str, Any]]]:
        # todo: improve, e.g. tuple[dict[str, tuple[np.array, np.array]], dict[str, tuple[np.array, np.array]]]
        """
        Output is produced over the whole of X

        :param should_plot_base_results:
        :param should_show_plots:
        :param skip_base_model_copy:
        :param should_save_plots:
        :param quantiles:
        :param should_plot_data:
        :param should_plot_uq_results:
        :return: UQ metrics and UQ results if return_results is False, else UQ metrics also native and posthoc results
        """
        # todo: bring back return_results?
        #  :param return_results: return native and posthoc results in addition to the native and posthoc metrics?

        print("loading data...")
        X_train, X_test, y_train, y_test, X, y, scaler_y = self.get_data()

        print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        if should_plot_data:
            print("plotting data...")
            self.plot_data(
                X_train,
                y_train,
                X_test,
                y_test,
                show_plot=should_show_plots,
                save_plot=should_save_plots,
            )

        print("training base models...")
        X_pred, y_true = X, y

        base_models = self.train_base_models(X_train, y_train)  # todo: what to do if empty?
        y_preds_base_models = self.predict_base_models(base_models, X_pred)

        if should_plot_base_results:
            print("plotting base model results...")
            for base_model_name, y_pred_base_model in y_preds_base_models.items():
                self.plot_base_results(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    y_pred_base_model,
                    plot_name=base_model_name,
                    show_plots=should_show_plots,
                    save_plot=should_save_plots,
                )

        print("computing base model metrics...")
        base_models_metrics = {
            model_name: self.compute_metrics_det(model_preds, y_true)
            for model_name, model_preds in y_preds_base_models.items()
        }
        self.print_base_models_metrics(base_models_metrics)

        print("running posthoc UQ methods...")
        posthoc_results = self.run_posthoc_methods(X_train, y_train, X_pred, base_models, quantiles=quantiles,
                                                   skip_base_model_copy=skip_base_model_copy)
        partial_plotting = partial(
            self.plot_uq_results,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            quantiles=quantiles,
            show_plots=should_show_plots,
            save_plots=should_save_plots,
        )
        if should_plot_uq_results:
            print("plotting posthoc results...")
            partial_plotting(posthoc_results)

        print("running native UQ methods...")
        native_results = self.run_native_methods(X_train, y_train, X_pred, quantiles=quantiles)

        if should_plot_uq_results:
            print("plotting native results...")
            partial_plotting(native_results)

        print("computing and saving UQ metrics...")
        uq_results_all = {'posthoc': posthoc_results, 'native': native_results}
        uq_metrics_all = {'base_model': base_models_metrics}
        for uq_type, uq_results in uq_results_all.items():
            print(f'{uq_type}...')
            uq_metrics_all[uq_type] = self.compute_all_metrics(uq_results, y_true, quantiles=quantiles)
        self.print_uq_metrics(uq_metrics_all)
        self.io_helper.save_metrics(uq_metrics_all, filename='uq')
        return base_models_metrics, uq_metrics_all

    @abstractmethod
    def get_data(self) -> DataTuple | Any:
        """

        :return: tuple (X_train, X_test, y_train, y_test, X, y, y_scaler). y_scaler may be None.
        """
        raise NotImplementedError

    def compute_metrics(
            self,
            y_pred,
            y_quantiles: np.ndarray | None,
            y_std: np.ndarray | None,
            y_true,
            quantiles=None,
    ) -> dict[str, float]:
        """
        Evaluate a UQ method by computing all desired metrics.

        :param y_pred:
        :param y_quantiles: array of shape (n_samples, n_quantiles) containing the predicted quantiles of y_true
        :param y_std: 1D array-like of predicted standard deviations around y_true
        :param y_true:
        :param quantiles: list of quantiles with which test to measure performance. They are expected to be symmetric
        :return:
        """
        metrics = self.compute_metrics_det(y_pred, y_true)
        metrics_uq = self.compute_metrics_uq(y_pred, y_quantiles, y_std, y_true, quantiles)
        metrics.update(metrics_uq)
        return metrics

    @abstractmethod
    def compute_metrics_det(
            self,
            y_pred,
            y_true,
    ) -> dict[str, float]:
        """
        Evaluate a UQ method by computing all desired deterministic metrics.

        :param y_pred:
        :param y_true:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def compute_metrics_uq(
            self,
            y_pred,
            y_quantiles: np.ndarray | None,
            y_std: np.ndarray | None,
            y_true,
            quantiles,
    ) -> dict[str, float]:
        """
        Evaluate a UQ method by computing all desired uncertainty metrics.

        :param y_pred:
        :param y_quantiles: array of shape (n_samples, n_quantiles) containing the predicted quantiles of y_true
        :param y_std: 1D array-like of predicted standard deviations around y_true
        :param y_true:
        :param quantiles: list of quantiles with which test to measure performance. They are expected to be symmetric
        :return:
        """
        raise NotImplementedError

    def compute_all_metrics(
            self,
            uq_results: dict[
                str,
                tuple[
                    np.ndarray,
                    np.ndarray | None,
                    np.ndarray | None
                ],
            ],
            y_true,
            quantiles=None,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate all UQ methods by computing all desired metrics.

        :param quantiles: quantiles
        :param uq_results: dict of (method_name, (y_pred, y_quantiles, y_std))
        :param y_true:
        :return: dict of { method_name: {metric_name: value,  ...}, ... }
        """
        return {
            method_name: self.compute_metrics(y_pred, y_quantiles, y_std, y_true, quantiles=quantiles)
            for method_name, (y_pred, y_quantiles, y_std) in uq_results.items()
        }

    def train_base_models(self, X_train, y_train) -> dict[str, Any]:
        """

        :param X_train:
        :param y_train:
        :return: dict of (base_model_name, base_model)
        """
        base_models_methods = self.get_base_model_methods()
        base_models = {}
        for method_name, method in base_models_methods:
            if self.method_whitelist is not None and method_name not in self.method_whitelist:
                continue
            base_model_kwargs = self.methods_kwargs[method_name]
            base_model = method(X_train, y_train, **base_model_kwargs)
            base_models[method_name] = base_model
        return base_models

    @staticmethod
    def predict_base_models(base_models: dict[str, Any], X_pred):
        base_model_preds = {}
        for model_name, base_model in base_models.items():
            base_model_preds[model_name] = base_model.predict(X_pred)
        return base_model_preds

    def get_base_model_methods(self):
        return self._get_methods_by_prefix('base_model')

    def get_posthoc_methods(self):
        return self._get_methods_by_prefix("posthoc")

    def get_native_methods(self):
        return self._get_methods_by_prefix("native")

    def _get_methods_by_prefix(self, prefix: str, sep='_') -> Generator[tuple[str, Callable], None, None]:
        """
        get all instance methods (i.e. callable attributes) with given prefix
        :param prefix:
        :return: generator of (method_name, method) pairs
        """
        full_prefix = prefix + sep
        for attr_name in self.__class__.__dict__.keys():
            attr = getattr(self, attr_name)
            if attr_name.startswith(full_prefix) and callable(attr):
                yield attr_name, attr

    def run_posthoc_methods(
            self,
            X_train,
            y_train,
            X_pred,
            base_models: dict[str, Any],
            quantiles,
            skip_base_model_copy=False,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """

        :param quantiles:
        :param base_models:
        :param skip_base_model_copy: whether to skip making a deepcopy of the base model. speeds up execution, but can
         lead to bugs if posthoc method affects the base model object.
        :param X_train:
        :param y_train:
        :param X_pred:
        :return: dict of (method_name, (y_pred, y_quantiles)), where y_pred and y_quantiles are 1D and 2D, respectively
        """
        posthoc_methods = self.get_posthoc_methods()
        if self.method_whitelist is not None:
            posthoc_methods = list(starfilter(lambda name, _: name in self.method_whitelist,
                                              posthoc_methods))
        if not posthoc_methods:
            print(f'No posthoc methods found and/or whitelisted. Skipping...')
            return dict()

        print(f"running posthoc methods...")
        posthoc_results = {}
        for posthoc_method_name, posthoc_method in posthoc_methods:
            blacklist = self.posthoc_base_blacklist[posthoc_method_name]
            compatible_base_models = {
                base_model_name: base_model
                for base_model_name, base_model in base_models.items()
                if base_model_name not in blacklist
            }
            if not compatible_base_models:
                print(f'no compatible base models found for posthoc method {posthoc_method_name} - skipping.')
                continue
            print(f'running {posthoc_method_name}...')

            method_kwargs = self.methods_kwargs[posthoc_method_name]
            for base_model_name, base_model in compatible_base_models.items():
                print(f'...on {base_model_name}...')
                base_model_copy = base_model if skip_base_model_copy else copy.deepcopy(base_model)
                y_pred, y_quantiles, y_std = posthoc_method(
                    X_train,
                    y_train,
                    X_pred,
                    quantiles,
                    base_model_copy,
                    **method_kwargs
                )
                key = f'{posthoc_method_name}__{base_model_name}'
                posthoc_results[key] = y_pred, y_quantiles, y_std
        return posthoc_results

    def run_native_methods(
            self,
            X_train,
            y_train,
            X_pred,
            quantiles,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """

        :param quantiles:
        :param X_train:
        :param y_train:
        :param X_pred:
        :return: dict of (method_name, y_pred), where y_pred is a 1D array
        """
        native_methods = self.get_native_methods()
        if self.method_whitelist is not None:
            native_methods = list(starfilter(lambda name, _: name in self.method_whitelist,
                                             native_methods))
        if not native_methods:
            print(f'No native methods found and/or whitelisted. Skipping...')
            return dict()

        print(f"running native methods...")
        native_results = {}
        for native_method_name, native_method in native_methods:
            print(f'running {native_method_name}...')
            method_kwargs = self.methods_kwargs[native_method_name]
            y_pred, y_quantiles, y_std = native_method(
                X_train,
                y_train,
                X_pred,
                quantiles=quantiles,
                **method_kwargs
            )
            native_results[native_method_name] = y_pred, y_quantiles, y_std
        return native_results

    @staticmethod
    def stds_from_quantiles(quantiles: np.ndarray):
        """
        :param quantiles: array of shape (number of datapoints, number of quantiles), where number of quantiles should
        be at least about 100
        :return:
        """
        num_quantiles = quantiles.shape[1]
        if num_quantiles < 50:
            print(f"warning: {num_quantiles} quantiles are too few"
                  f" to compute a reliable std from (should be about 100)")
        return np.std(quantiles, ddof=1, axis=1)

    @staticmethod
    def pis_from_quantiles(quantiles):
        mid = len(quantiles) // 2
        first, second = quantiles[:mid], quantiles[mid:]
        pi_limits = zip(first, reversed(second))
        pis = [high - low for low, high in pi_limits]
        return sorted(pis)

    @staticmethod
    def quantiles_from_pis(pis: np.ndarray, check_order=False):
        """
        currently "buggy" for odd number of quantiles.
        :param check_order:
        :param pis: prediction intervals array of shape (n_samples, 2, n_intervals)
        :return: array of quantiles of shape (n_samples, 2 * n_intervals)
        """
        # todo: assumption that quantile ordering is definitely consistent fulfilled?
        if check_order:
            from helpers import is_ascending
            assert np.all([is_ascending(pi[0, :], reversed(pi[1, :])) for pi in pis])
        y_quantiles = np.array([sorted(pi.flatten()) for pi in pis])
        return y_quantiles

    #### PLOTTING ###

    def plot_data(
            self,
            X_train,
            y_train,
            X_test,
            y_test,
            filename="data",
            figsize=(16, 5),
            ylabel="energy data",
            show_plot=True,
            save_plot=True,
    ):
        """visualize training and test sets"""
        from matplotlib import pyplot as plt

        num_train_steps = X_train.shape[0]
        num_test_steps = X_test.shape[0]

        x_plot_train = np.arange(num_train_steps)
        x_plot_test = x_plot_train + num_test_steps

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_plot_train, y_train)
        ax.plot(x_plot_test, y_test)
        ax.set_ylabel(ylabel)
        ax.legend(["Training data", "Test data"])
        if save_plot:
            self.io_helper.save_plot(filename)
        if show_plot:
            plt.show(block=True)
        plt.close(fig)

    def plot_uq_results(
            self,
            X_train,
            y_train,
            X_test,
            y_test,
            uq_results: dict[str, tuple[Any, Any, Any]],
            quantiles,
            show_plots=True,
            save_plots=True,
            n_stds=2,
    ):
        # todo: allow results to have multiple PIs (corresp. to multiple alphas)?
        for method_name, (y_preds, y_quantiles, y_std) in uq_results.items():
            if y_quantiles is None and y_std is None:
                print(f"warning: cannot plot method {method_name}, because both y_quantiles and y_std are None")
                continue
            self.plot_uq_result(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                y_preds=y_preds,
                y_quantiles=y_quantiles,
                y_std=y_std,
                quantiles=quantiles,
                n_stds=n_stds,
                plot_name=method_name,
                show_plots=show_plots,
                save_plot=save_plots,
            )

    def plot_uq_result(
            self,
            X_train,
            y_train,
            X_test,
            y_test,
            y_preds,
            y_quantiles,
            y_std,
            quantiles,
            n_stds=2,
            plot_name='uq_result',
            show_plots=True,
            save_plot=True,
    ):
        from matplotlib import pyplot as plt
        num_train_steps, num_test_steps = X_train.shape[0], X_test.shape[0]

        x_plot_train = np.arange(num_train_steps)
        x_plot_test = np.arange(num_train_steps, num_train_steps + num_test_steps)
        x_plot_full = np.arange(num_train_steps + num_test_steps)

        drawing_quantiles = y_quantiles is not None
        if drawing_quantiles:
            ci_low, ci_high = (
                y_quantiles[:, 0],
                y_quantiles[:, -1],
            )
            drawn_quantile = round(max(quantiles) - min(quantiles), 2)
        else:
            ci_low, ci_high = y_preds - n_stds * y_std, y_preds + n_stds * y_std

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
        ax.plot(x_plot_train, y_train, label='y_train', linestyle="dashed", color="black")
        ax.plot(x_plot_test, y_test, label='y_test', linestyle="dashed", color="blue")
        ax.plot(
            x_plot_full,
            y_preds,
            label=f"mean/median prediction",  # todo: mean or median?
            color="green",
        )
        # noinspection PyUnboundLocalVariable
        label = rf'{100 * drawn_quantile}% CI' if drawing_quantiles else f'{n_stds} std'
        ax.fill_between(
            x_plot_full.ravel(),
            ci_low,
            ci_high,
            color="green",
            alpha=0.2,
            label=label,
        )
        ax.legend()
        ax.set_xlabel("data")
        ax.set_ylabel("target")
        ax.set_title(plot_name)
        if save_plot:
            self.io_helper.save_plot(plot_name)
        if show_plots:
            plt.show(block=True)
        plt.close(fig)

    def plot_base_results(
            self,
            X_train,
            y_train,
            X_test,
            y_test,
            y_preds,
            plot_name='base_results',
            show_plots=True,
            save_plot=True,
    ):
        from matplotlib import pyplot as plt
        num_train_steps, num_test_steps = X_train.shape[0], X_test.shape[0]

        x_plot_train = np.arange(num_train_steps)
        x_plot_full = np.arange(num_train_steps + num_test_steps)
        x_plot_test = np.arange(num_train_steps, num_train_steps + num_test_steps)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
        ax.plot(x_plot_train, y_train, label='y_train', linestyle="dashed", color="black")
        ax.plot(x_plot_test, y_test, label='y_test', linestyle="dashed", color="blue")
        ax.plot(
            x_plot_full,
            y_preds,
            label=f"mean/median prediction {plot_name}",  # todo: mean or median?
            color="green",
        )
        ax.legend()
        ax.set_xlabel("data")
        ax.set_ylabel("target")
        ax.set_title(plot_name)
        if save_plot:
            self.io_helper.save_plot(plot_name)
        if show_plots:
            plt.show(block=True)
        plt.close(fig)

    @staticmethod
    def print_uq_metrics(uq_metrics):
        print()
        for uq_type, method_metrics in uq_metrics.items():
            print(f"{uq_type} metrics:")
            for method, metrics in method_metrics.items():
                print(f"\t{method}:")
                for metric, value in metrics.items():
                    print(f"\t\t{metric}: {value}")
            print()

    @staticmethod
    def print_base_models_metrics(base_models_metrics):
        print()
        for model_name, metrics in base_models_metrics.items():
            print(f"{model_name}:")
            for metric, value in metrics.items():
                print(f"\t{metric}: {value}")
        print()


def check_prefixes_ok():
    forbidden_prefixes = ['native', 'posthoc', 'base_model']
    for attr_name in UQ_Comparer.__dict__.keys():
        for prefix in forbidden_prefixes:
            assert not attr_name.startswith(prefix)
    print('all prefixes ok')


if __name__ == '__main__':
    check_prefixes_ok()
