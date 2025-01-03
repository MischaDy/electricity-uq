import logging
import os

filename = os.path.split(__file__)[-1]
logging.info(f'reading file {filename}...')

from abc import ABC, abstractmethod
import numpy as np
import copy
from functools import partial
from typing import Any, Generator, Callable, TYPE_CHECKING

from helpers.io_helper import IO_Helper
from helpers import misc_helpers

if TYPE_CHECKING:
    from helpers.typing_ import UQ_Output


# noinspection PyPep8Naming
class UQ_Comparison_Pipeline_ABC(ABC):
    # todo: update usage guide!
    """
    Usage:
    1. Inherit from this class.
    2. Override the get_data and compute_metrics methods.
    3. Define all desired point estimator ('deterministic') methods. The required signature is:
            (X_train, y_train, X_val, y_val, ...) -> trained_model
       The methods' names should start with 'base_model_' and they should be instance methods.
    4. Define all desired native UQ methods. The required signature is:
            (X_train, y_train, X_val, y_val, X_test, quantiles, ...) -> (y_pred, y_quantiles, y_std)
       y_quantiles is ... . y_std is ... .
       The methods' names should start with 'native_' and they should be instance methods.
    5. Define all desired posthoc UQ methods. The required signature is:
            (X_train, y_train, X_val, y_val, X_test, quantiles, base_model, ...) -> (y_pred, y_quantiles, y_std)
       y_quantiles and y_std are the same as for native methods.
       The methods' names should start with 'posthoc_' and they should be instance methods.
    6. Call compare_methods from the child class.
    """

    # todo: for child classes to override
    posthoc_base_blacklist: dict[str, set[str]]  # dict of pairs (posthoc_method_name, incompatible_base_methods set)

    def __init__(
            self,
            storage_path,
            data_path,
            methods_kwargs,
            filename_parts,
            method_whitelist=None,
            do_standardize_data=True,
    ):
        """
        :param storage_path:
        :param data_path:
        :param filename_parts: see IO_Helper.filename_parts definition
        :param method_whitelist:
        :param do_standardize_data: True if both X and y should be standardized, False if neither.
        """
        # todo: store train and test data once loaded?
        self.data_path = data_path
        self.methods_kwargs = methods_kwargs
        self.io_helper = IO_Helper(storage_path, methods_kwargs=methods_kwargs, filename_parts=filename_parts)
        self.method_whitelist = method_whitelist
        self.do_standardize_data = do_standardize_data

    def compare_methods(
            self,
            quantiles: list,
            *,
            should_save_results=True,
            should_plot_data=True,
            should_plot_uq_results=True,
            should_plot_base_results=True,
            should_show_plots=True,
            should_save_plots=True,
            skip_base_model_copy=False,
            use_filesave_prefix=True,
    ) -> tuple[dict, dict] | dict[str, dict[str, dict[str, Any]]]:
        """
        ...
        Output is produced over the whole of X!

        :param use_filesave_prefix: if True, save files with prefix "n{number of samples}"
        :param should_save_results:
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
        logging.info("loading data...")

        assert 0.5 in quantiles
        X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = self.get_data()
        if use_filesave_prefix:
            self.io_helper.filesave_prefix = f'n{X_train.shape[0]}'

        logging.info(f"data shapes: {X_train.shape}, {X_val.shape}, {X_test.shape};"
                     f"  {y_train.shape}, {y_val.shape}, {y_test.shape}")
        if should_plot_data:
            logging.info("plotting data...")
            self.plot_data(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                scaler_y=scaler_y,
                show_plot=should_show_plots,
                save_plot=should_save_plots,
            )

        logging.info("training base models...")
        X_pred, y_true = X, y
        y_true_orig_scale = misc_helpers.inverse_transform_y(scaler_y, y_true)

        base_models = self.train_base_models(X_train, y_train, X_val, y_val)  # todo: what to do if empty?
        y_preds_base_models = self.predict_base_models(base_models, X_pred, scaler_y=scaler_y)
        if should_save_results:
            logging.info('saving base model results...')
            self.save_outputs_base_models(y_preds_base_models)
        else:
            logging.info('base model result saving is skipped.')

        if should_plot_base_results:
            logging.info("plotting base model results...")
            for base_model_name, y_pred_base_model in y_preds_base_models.items():
                self.plot_base_results(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    y_pred_base_model,
                    scaler_y=scaler_y,
                    base_model_name=base_model_name,
                    show_plots=should_show_plots,
                    save_plot=should_save_plots,
                )

        logging.info("computing base model metrics...")
        base_models_metrics = {
            model_name: self.compute_metrics_det(model_preds, y_true_orig_scale)
            for model_name, model_preds in y_preds_base_models.items()
        }
        if base_models_metrics:
            self.print_base_models_metrics(base_models_metrics)

        logging.info("running posthoc UQ methods...")
        posthoc_results = self.run_posthoc_methods(
            X_train,
            y_train,
            X_val,
            y_val,
            X_pred,
            base_models,
            quantiles=quantiles,
            scaler_y=scaler_y,
            skip_base_model_copy=skip_base_model_copy
        )
        if should_save_results:
            logging.info('saving posthoc UQ results...')
            self.save_outputs_uq_models(posthoc_results)
        else:
            logging.info('posthoc UQ results saving is skipped.')

        plot_uq_results = partial(
            self.plot_uq_results,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            quantiles=quantiles,
            scaler_y=scaler_y,
            show_plots=should_show_plots,
            save_plots=should_save_plots,
        )
        if should_plot_uq_results:
            logging.info("plotting posthoc results...")
            plot_uq_results(uq_results=posthoc_results)

        logging.info("running native UQ methods...")
        native_results = self.run_native_methods(
            X_train,
            y_train,
            X_val,
            y_val,
            X_pred,
            quantiles=quantiles,
            scaler_y=scaler_y,
        )

        if should_save_results:
            logging.info('saving native UQ results...')
            self.save_outputs_uq_models(native_results)
        else:
            logging.info('native UQ result saving is skipped.')

        if should_plot_uq_results:
            logging.info("plotting native results...")
            plot_uq_results(uq_results=native_results)

        logging.info("computing and saving UQ metrics...")
        uq_results_all = {'posthoc': posthoc_results, 'native': native_results}
        uq_metrics_all = {'base_model': base_models_metrics}
        for uq_type, uq_results in uq_results_all.items():
            logging.info(f'{uq_type}...')
            uq_metrics_all[uq_type] = self.compute_all_metrics(uq_results, y_true_orig_scale, quantiles=quantiles)
        self.print_uq_metrics(uq_metrics_all, print_optimal=True)
        self.io_helper.save_metrics(uq_metrics_all, filename='uq_metrics')
        return base_models_metrics, uq_metrics_all

    @abstractmethod
    def get_data(self):
        """

        :return: tuple (X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y). y_scaler may be None.
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
            uq_results: dict[str, 'UQ_Output'],
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

    def train_base_models(self, X_train, y_train, X_val, y_val) -> dict[str, Any]:
        """

        :param y_val:
        :param X_val:
        :param X_train:
        :param y_train:
        :return: dict of (base_model_name, base_model)
        """
        base_models_methods = self.get_base_model_methods()
        base_models = {}
        for method_name, method in base_models_methods:
            logging.info(f'training {method_name}...')
            if self.method_whitelist is not None and method_name not in self.method_whitelist:
                continue
            base_model_kwargs = self.methods_kwargs[method_name]
            base_model = method(X_train, y_train, X_val, y_val, **base_model_kwargs)
            base_models[method_name] = base_model
        return base_models

    @staticmethod
    def predict_base_models(base_models: dict[str, Any], X_pred, scaler_y=None):
        base_model_preds = {}
        for model_name, base_model in base_models.items():
            y_pred = base_model.predict(X_pred)
            if scaler_y is not None:
                y_pred = misc_helpers.inverse_transform_y(scaler_y, y_pred)
            base_model_preds[model_name] = y_pred
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
            X_val,
            y_val,
            X_pred,
            base_models: dict[str, Any],
            quantiles,
            scaler_y=None,
            skip_base_model_copy=False,
    ) -> dict[str, 'UQ_Output']:
        """

        :param y_val:
        :param X_val:
        :param scaler_y: sklearn-like scaler fitted on y_train with an inverse_transform method
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
            posthoc_methods = list(misc_helpers.starfilter(lambda name, _: name in self.method_whitelist,
                                                           posthoc_methods))
        if not posthoc_methods:
            logging.info(f'No posthoc methods found and/or whitelisted. Skipping...')
            return dict()

        logging.info(f"running posthoc methods...")
        posthoc_results = {}
        for posthoc_method_name, posthoc_method in posthoc_methods:
            blacklist = self.posthoc_base_blacklist[posthoc_method_name]
            compatible_base_models = {
                base_model_name: base_model
                for base_model_name, base_model in base_models.items()
                if base_model_name not in blacklist
            }
            if not compatible_base_models:
                logging.info(f'no compatible base models found for posthoc method {posthoc_method_name} - skipping.')
                continue

            method_kwargs = self.methods_kwargs[posthoc_method_name]
            for base_model_name, base_model in compatible_base_models.items():
                logging.info(f'running {posthoc_method_name} on {base_model_name}...')
                base_model_copy = base_model if skip_base_model_copy else copy.deepcopy(base_model)
                y_pred, y_quantiles, y_std = posthoc_method(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_pred,
                    quantiles,
                    base_model_copy,
                    **method_kwargs
                )
                if scaler_y is not None:
                    y_pred = misc_helpers.inverse_transform_y(scaler_y, y_pred)
                    if y_quantiles is not None:
                        y_quantiles = misc_helpers.inverse_transform_ys(scaler_y, *y_quantiles, to_np=True)
                    if y_std is not None:
                        y_std = misc_helpers.upscale_y_std(scaler_y, y_std)
                key = f'{posthoc_method_name}__{base_model_name}'
                posthoc_results[key] = y_pred, y_quantiles, y_std
        return posthoc_results

    def run_native_methods(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            X_pred,
            quantiles,
            scaler_y=None,
    ) -> dict[str, 'UQ_Output']:
        """

        :param y_val:
        :param X_val:
        :param scaler_y: sklearn-like scaler fitted on y_train with an inverse_transform method
        :param quantiles:
        :param X_train:
        :param y_train:
        :param X_pred:
        :return: dict of (method_name, y_pred), where y_pred is a 1D array
        """
        native_methods = self.get_native_methods()
        if self.method_whitelist is not None:
            native_methods = list(misc_helpers.starfilter(lambda name, _: name in self.method_whitelist,
                                                          native_methods))
        if not native_methods:
            logging.info(f'No native methods found and/or whitelisted. Skipping...')
            return dict()

        logging.info(f"running native methods...")
        native_results = {}
        for native_method_name, native_method in native_methods:
            logging.info(f'running {native_method_name}...')
            method_kwargs = self.methods_kwargs[native_method_name]
            y_pred, y_quantiles, y_std = native_method(
                X_train,
                y_train,
                X_val,
                y_val,
                X_pred,
                quantiles=quantiles,
                **method_kwargs
            )
            if scaler_y is not None:
                y_pred = misc_helpers.inverse_transform_y(scaler_y, y_pred)
                if y_quantiles is not None:
                    y_quantiles = misc_helpers.inverse_transform_ys(scaler_y, *y_quantiles, to_np=True)
                if y_std is not None:
                    y_std = misc_helpers.upscale_y_std(scaler_y, y_std)
            native_results[native_method_name] = y_pred, y_quantiles, y_std
        return native_results

    #### PLOTTING ###

    def plot_data(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            scaler_y=None,
            filename="data",
            figsize=(16, 5),
            ylabel="energy demand",
            show_plot=True,
            save_plot=True,
    ):
        """
        visualize training and test sets

        :param y_val:
        :param X_val:
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param scaler_y: sklearn-like scaler fitted on y_train with an inverse_transform method
        :param filename:
        :param figsize:
        :param ylabel:
        :param show_plot:
        :param save_plot:
        :return:
        """
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        self._plot_data_worker(X_test, X_train, X_val, y_test, y_train, y_val, ax, scaler_y=scaler_y)
        ax.set_ylabel(ylabel)
        ax.legend()
        if save_plot:
            self.io_helper.save_plot(filename=filename)
        if show_plot:
            plt.show(block=True)
        plt.close(fig)

    @staticmethod
    def _plot_data_worker(X_train, y_train, X_val, y_val, X_test, y_test, ax, scaler_y=None, linestyle="dashed"):
        start_train, end_train = 0, X_train.shape[0]
        start_val, end_val = end_train, end_train + X_val.shape[0]
        start_test, end_test = end_val, end_val + X_test.shape[0]

        x_plot_train = np.arange(start_train, end_train)
        x_plot_val = np.arange(start_val, end_val)
        x_plot_test = np.arange(start_test, end_test)
        if scaler_y is not None:
            y_train, y_val, y_test = misc_helpers.inverse_transform_ys(scaler_y, y_train, y_val, y_test)

        ax.plot(x_plot_train, y_train, label='train data', color="black", linestyle=linestyle)
        ax.plot(x_plot_val, y_val, label='val data', color="purple", linestyle=linestyle)  # todo: violet?
        ax.plot(x_plot_test, y_test, label='test data', color="blue", linestyle=linestyle)

    @staticmethod
    def _get_x_plot_full(X_train, X_val, X_test):
        num_train_steps = X_train.shape[0]
        num_val_steps = X_val.shape[0]
        num_test_steps = X_test.shape[0]
        num_steps = num_train_steps + num_val_steps + num_test_steps
        return np.arange(num_steps)

    def plot_uq_results(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            uq_results: dict[str, 'UQ_Output'],
            quantiles,
            scaler_y=None,
            show_plots=True,
            save_plots=True,
            n_stds=2,
    ):
        """

        :param y_val:
        :param X_val:
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param scaler_y: sklearn-like scaler fitted on y_train with an inverse_transform method
        :param uq_results:
        :param quantiles:
        :param show_plots:
        :param save_plots:
        :param n_stds:
        :return:
        """
        for method_name, (y_pred, y_quantiles, y_std) in uq_results.items():
            if y_quantiles is None and y_std is None:
                logging.warning(f"cannot plot method {method_name}, because both y_quantiles and y_std are None")
                continue
            self.plot_uq_result(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                y_pred=y_pred,
                y_quantiles=y_quantiles,
                y_std=y_std,
                quantiles=quantiles,
                method_name=method_name,
                scaler_y=scaler_y,
                n_stds=n_stds,
                show_plots=show_plots,
                save_plot=save_plots,
            )

    def plot_uq_result(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_pred,
            y_quantiles,
            y_std,
            quantiles,
            method_name,
            scaler_y=None,
            n_stds=2,
            show_plots=True,
            save_plot=True,
    ):
        """

        :param y_val:
        :param X_val:
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param y_pred:
        :param y_quantiles:
        :param y_std:
        :param scaler_y: sklearn-like scaler fitted on y_train with an inverse_transform method
        :param quantiles:
        :param n_stds:
        :param method_name:
        :param show_plots:
        :param save_plot:
        :return:
        """
        from matplotlib import pyplot as plt

        drawing_quantiles = y_quantiles is not None
        if drawing_quantiles:
            ci_low, ci_high = (
                y_quantiles[:, 0],
                y_quantiles[:, -1],
            )
            drawn_quantile = round(max(quantiles) - min(quantiles), 2)
        else:
            ci_low, ci_high = y_pred - n_stds * y_std, y_pred + n_stds * y_std

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
        self._plot_data_worker(X_train, y_train, X_val, y_val, X_test, y_test, ax=ax, scaler_y=scaler_y)
        x_plot_full = self._get_x_plot_full(X_train, X_val, X_test)
        ax.plot(x_plot_full, y_pred, label="point prediction", color="green")
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
        ax.set_title(method_name)
        if save_plot:
            self.io_helper.save_plot(method_name=method_name)
        if show_plots:
            plt.show(block=True)
        plt.close(fig)

    def plot_base_results(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            y_pred,
            base_model_name,
            scaler_y=None,
            show_plots=True,
            save_plot=True,
    ):
        """

        :param y_val:
        :param X_val:
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param y_pred:
        :param scaler_y: sklearn-like scaler fitted on y_train with an inverse_transform method
        :param base_model_name:
        :param show_plots:
        :param save_plot:
        :return:
        """
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
        self._plot_data_worker(X_train, y_train, X_val, y_val, X_test, y_test, ax, scaler_y=scaler_y)
        x_plot_full = self._get_x_plot_full(X_train, X_val, X_test)
        ax.plot(x_plot_full, y_pred, label="point prediction", color="green")
        ax.legend()
        ax.set_xlabel("data")
        ax.set_ylabel("target")
        ax.set_title(base_model_name)
        if save_plot:
            self.io_helper.save_plot(method_name=base_model_name)
        if show_plots:
            plt.show(block=True)
        plt.close(fig)

    def print_uq_metrics(self, uq_metrics, print_optimal=True, y_true_orig_scale=None, y_quantiles=None, y_std=None,
                         quantiles=None, eps_std=1e-2):
        print()
        for uq_type, method_metrics in uq_metrics.items():
            print(f"{uq_type} metrics:")
            for method, metrics in method_metrics.items():
                print(f"\t{method}:")
                for metric, value in metrics.items():
                    print(f"\t\t{metric}: {value}")
            print()
        if print_optimal:
            self.print_optimal_det_metrics(y_true_orig_scale)
            self.print_optimal_uq_metrics(y_true_orig_scale, y_quantiles, y_std, quantiles, eps_std=eps_std)

    def print_optimal_uq_metrics(self, y_true_orig_scale, y_quantiles, y_std, quantiles, eps_std=1e-2):
        print('\toptimal uq metrics:')
        y_quantiles_opt = np.hstack([y_true_orig_scale] * y_quantiles.shape[1])
        y_std_opt = np.ones_like(y_std) * eps_std
        optimal_uq_metrics = self.compute_metrics_uq(y_true_orig_scale, y_quantiles_opt, y_std_opt, y_true_orig_scale,
                                                     quantiles=quantiles)
        for metric, value in optimal_uq_metrics.items():
            print(f"\t\t{metric}: {value}")

    def print_base_models_metrics(self, base_models_metrics, print_optimal=True, y_true_orig_scale=None):
        print()
        for model_name, metrics in base_models_metrics.items():
            print(f"{model_name}:")
            for metric, value in metrics.items():
                print(f"\t{metric}: {value}")
        print()
        if print_optimal:
            self.print_optimal_det_metrics(y_true_orig_scale)

    def print_optimal_det_metrics(self, y_true_orig_scale):
        optimal_det_metrics = self.compute_metrics_det(y_true_orig_scale, y_true_orig_scale)
        print('optimal deterministic metrics:')
        for metric, value in optimal_det_metrics.items():
            print(f"\t\t{metric}: {value}")

    def save_outputs_base_models(self, y_preds_dict: dict[str, np.ndarray]):
        for base_model_name, y_pred in y_preds_dict.items():
            self.io_helper.save_array(y_pred, method_name=base_model_name)

    def save_outputs_uq_models(self, outputs_uq_models: dict[str, 'UQ_Output']):
        for model_name, outputs_uq_model in outputs_uq_models.items():
            for output_type, output in zip(['y_pred', 'y_quantiles', 'y_std'], outputs_uq_model):
                self.io_helper.save_array(output, method_name=model_name, infix=output_type)


def check_prefixes_ok():
    forbidden_prefixes = ['native', 'posthoc', 'base_model']
    for attr_name in UQ_Comparison_Pipeline_ABC.__dict__.keys():
        for prefix in forbidden_prefixes:
            assert not attr_name.startswith(prefix)
    logging.info('all prefixes ok')


if __name__ == '__main__':
    check_prefixes_ok()
