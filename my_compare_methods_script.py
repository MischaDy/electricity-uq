import time
from typing import Any

import numpy as np
import numpy.typing as npt

import pandas as pd
import skorch.callbacks
from mapie.subsample import BlockBootstrap
from matplotlib import pyplot as plt
from more_itertools import collapse
from pmdarima.metrics import smape

from scipy.stats import randint, norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from skorch import NeuralNetRegressor
from skorch.dataset import Dataset
from statsmodels.tools.eval_measures import rmse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from uncertainty_toolbox.metrics_scoring_rule import nll_gaussian

# from properscoring import crps_ensemble

from compare_methods import UQ_Comparer
from helpers import get_data, IO_Helper, standardize

from conformal_prediction import estimate_pred_interals_no_pfit_enbpi
from quantile_regression import estimate_quantiles as estimate_quantiles_qr

import torch
from torch import nn

from laplace import Laplace

from temp_nn_file import My_NN
from temp_sklearn_estimator import MyEstimator

METHOD_WHITELIST = [
    "posthoc_conformal_prediction",
    # "posthoc_laplace",
    # "native_quantile_regression",
    # "native_gp",
]
TO_STANDARDIZE = "xy"
QUANTILES = [
    0.05,
    0.25,
    0.75,
    0.95,
]  # todo: how to handle 0.5? ==> just use mean if needed

PLOT_DATA = False
PLOT_RESULTS = True
SAVE_PLOTS = True
SAVE_TRAINED = False

PLOTS_PATH = "plots"

BASE_MODEL_PARAMS = {
    "skip_training": False,
    # 'n_jobs': -1,
    # 'model_params_choices': None,
}

torch.set_default_dtype(torch.float32)


def print_metrics(uq_metrics: dict[str, dict[str, dict[str, Any]]]):
    print()
    for uq_type, method_metrics in uq_metrics.items():
        print(f"{uq_type} metrics:")
        for method, metrics in method_metrics.items():
            print(f"\t{method}:")
            for metric, value in metrics.items():
                print(f"\t\t{metric}: {value}")
        print()


# noinspection PyPep8Naming
class My_UQ_Comparer(UQ_Comparer):
    def __init__(self, storage_path="comparison_storage", to_standardize="X", *args, **kwargs):
        """

        :param storage_path:
        :param to_standardize: iterable of variables to standardize. Can contain 'x' and/or 'y', or neither.
        :param args: passed to super.__init__
        :param kwargs: passed to super.__init__
        """
        super().__init__(*args, **kwargs)
        self.io_helper = IO_Helper(storage_path)
        self.to_standardize = to_standardize

    # todo: remove param?
    def get_data(self, _n_points_per_group=800):
        """

        :param _n_points_per_group:
        :return: X_train, X_test, y_train, y_test, X, y
        """
        X_train, X_test, y_train, y_test, X, y = get_data(
            _n_points_per_group, return_full_data=True
        )
        X_train, X_test, X = self._standardize_or_to_array("x", X_train, X_test, X)
        y_train, y_test, y = self._standardize_or_to_array("y", y_train, y_test, y)
        return X_train, X_test, y_train, y_test, X, y

    def _standardize_or_to_array(self, variable, *dfs):
        if variable in self.to_standardize:
            return standardize(False, *dfs)
        return map(lambda df: df.to_numpy(dtype=float), dfs)

    # todo: type hints!
    def compute_metrics(
        self, y_pred, y_quantiles, y_std, y_true: npt.NDArray[float], quantiles=None
    ):
        """

        :param y_pred: predicted y-values
        :param y_quantiles:
        :param y_std:
        :param y_true:
        :param quantiles:
        :return:
        """
        y_true = y_true.squeeze()
        # todo: sharpness? calibration? PIT? coverage?
        # todo: skill score (but what to use as benchmark)?

        metrics = {  # todo: improve
            "rmse": rmse(y_true, y_pred),
            "smape": smape(y_true, y_pred) / 100,  # scale down to [0, 1]
            "crps": (
                # todo: implement
                None  # crps_ensemble(y_pred, y_std, y_true_np) if y_std is not None else None
            ),
            "neg_log_lik": (
                nll_gaussian(y_pred, y_std, y_true) if y_std is not None else None
            ),
            "mean_pinball": (
                self._mean_pinball_loss(y_pred, y_quantiles, quantiles)
                if y_quantiles is not None
                else None
            ),
        }
        return metrics

    @staticmethod
    def _mean_pinball_loss(y_true, y_quantiles, quantiles):
        """

        :param y_true:
        :param y_quantiles: array of shape (n_samples, n_quantiles)
        :param quantiles:
        :return:
        """
        # fmt: off
        return np.mean([
            mean_pinball_loss(y_true, y_quantiles[:, ind], alpha=quantile)
            for ind, quantile in enumerate(quantiles)
        ])

    def train_base_model(self, *args, **kwargs):
        # todo: more flexibility in choosing (multiple) base models
        # model = self.my_train_base_model_rf(*args, **kwargs)
        model = self.my_train_base_model_nn(*args, save_trained=SAVE_TRAINED, **kwargs)
        return model

    def my_train_base_model_rf(
        self,
        X_train: npt.NDArray[float],
        y_train: npt.NDArray[float],
        model_params_choices=None,
        model_init_params=None,
        skip_training=True,
        n_jobs=-1,
        cv_n_iter=10,
    ):
        # todo: more flexibility in choosing (multiple) base models
        if model_params_choices is None:
            model_params_choices = {
                "max_depth": randint(2, 30),
                "n_estimators": randint(10, 100),
            }
        random_state = 59
        if model_init_params is None:
            model_init_params = {}
        elif "random_state" not in model_init_params:
            model_init_params["random_state"] = random_state

        model_class = RandomForestRegressor
        filename_base_model = f"base_{model_class.__name__}.model"

        if skip_training:
            # Model previously optimized with a cross-validation:
            # RandomForestRegressor(max_depth=13, n_estimators=89, random_state=59)
            try:
                print('skipping base model training')
                model = self.io_helper.load_model(filename_base_model)
                return model
            except FileNotFoundError:
                print(f"trained base model '{filename_base_model}' not found")

        assert all(
            item is not None for item in [X_train, y_train, model_params_choices]
        )
        print("training")

        # CV parameter search
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        model = model_class(random_state=random_state, **model_init_params)
        cv_obj = RandomizedSearchCV(
            model,
            param_distributions=model_params_choices,
            n_iter=cv_n_iter,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=random_state,
            verbose=1,
            n_jobs=n_jobs,
        )
        # todo: ravel?
        cv_obj.fit(X_train, y_train.ravel())
        model = cv_obj.best_estimator_
        print("done")
        self.io_helper.save_model(model, filename_base_model)
        return model

    def my_train_base_model_nn(
        self,
        X_train: npt.NDArray[float],
        y_train: npt.NDArray[float],
        model_params_choices=None,
        n_iter=20,
        batch_size=20,
        random_state=711,
        verbose=True,
        skip_training=True,
        save_trained=True,
        model_filename="_laplace_base.pth",
        val_frac=0.1,
        lr=0.1,
        lr_patience=5,
        lr_reduction_factor=0.1,
    ):
        """

        :param val_frac:
        :param lr_reduction_factor:
        :param lr:
        :param lr_patience:
        :param model_filename:
        :param save_trained:
        :param skip_training:
        :param verbose:
        :param X_train: shape (n_samples, n_dims)
        :param y_train: shape (n_samples, n_dims)
        :param model_params_choices:
        :param n_iter:
        :param batch_size:
        :param random_state:
        :return:
        """
        if skip_training:
            print("skipping base model training")
            try:
                model = self.io_helper.load_model(model_filename)
                model.eval()
                return model
            except FileNotFoundError:
                # fmt: off
                print("error. model not found, so training cannot be skipped. training from scratch")

        model = MyEstimator(
            n_iter=n_iter,
            batch_size=batch_size,
            random_state=random_state,
            val_frac=val_frac,
            lr=lr,
            lr_patience=lr_patience,
            lr_reduction_factor=lr_reduction_factor,
            verbose=verbose,
        )
        model.fit(X_train, y_train)

        if verbose:
            self.plot_post_training_perf(model, X_train, y_train, do_save_figure=True)

        if save_trained:
            model_savepath = self.io_helper.get_model_savepath(model_filename)
            torch.save(model, model_savepath)

        # noinspection PyTypeChecker
        model.set_params(verbose=False)
        return model

    @staticmethod
    def _plot_training_progress(train_losses, test_losses):
        fig, ax = plt.subplots()
        ax.semilogy(train_losses, label="train")
        ax.semilogy(test_losses, label="val")
        ax.legend()
        plt.show()

    def plot_post_training_perf(self, base_model, X_train, y_train, do_save_figure=False, filename='base_model'):
        y_preds = base_model.predict(X_train)

        num_train_steps = X_train.shape[0]
        x_plot_train = np.arange(num_train_steps)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
        ax.plot(x_plot_train, y_train, label='y_train', linestyle="dashed", color="black")
        ax.plot(
            x_plot_train,
            y_preds,
            label=f"base model prediction",
            color="green",
        )
        ax.legend()
        ax.set_xlabel("data")
        ax.set_ylabel("target")
        if do_save_figure:
            self.io_helper.save_plot(f"{filename}.png")
        plt.show()

    @staticmethod
    def _mse_torch(y_pred, y_test):
        return torch.mean((y_pred - y_test) ** 2)

    def posthoc_conformal_prediction(
        self, X_train, y_train, X_uq, quantiles, model, random_state=42
    ):
        cv = BlockBootstrap(
            n_resamplings=10, n_blocks=10, overlapping=False, random_state=random_state
        )
        alphas = self.pis_from_quantiles(quantiles)
        y_pred, y_pis = estimate_pred_interals_no_pfit_enbpi(
            model,
            cv,
            alphas,
            X_uq,
            X_train,
            y_train,
            skip_training=False,
            io_helper=self.io_helper,
        )
        y_quantiles = self.quantiles_from_pis(y_pis)  # (n_samples, 2 * n_intervals)
        if 0.5 in quantiles:
            num_quantiles = y_quantiles.shape[-1]
            ind = num_quantiles / 2
            y_quantiles = np.insert(y_quantiles, ind, y_pred, axis=1)
        y_std = None  # self.stds_from_quantiles(y_quantiles)
        return y_pred, y_quantiles, y_std

    def posthoc_laplace(
        self,
        X_train: npt.NDArray[float],
        y_train: npt.NDArray[float],
        X_uq: npt.NDArray[float],
        quantiles,
        model,
        n_iter=100,
        batch_size=20,
        random_state=711,
        verbose=True,
    ):
        # todo: offer option to alternatively optimize parameters and hyperparameters of the prior jointly (cf. example
        #  script)?
        X_train, y_train = map(lambda arr: self._arr_to_tensor(arr), (X_train, y_train))
        train_loader = self._get_train_loader(X_train, y_train, batch_size)

        la = Laplace(model, "regression")
        la.fit(train_loader)
        log_prior, log_sigma = (
            torch.ones(1, requires_grad=True),
            torch.ones(1, requires_grad=True),
        )
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
        iterable = tqdm(range(n_iter)) if verbose else range(n_iter)
        for _ in iterable:
            hyper_optimizer.zero_grad()
            neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
            neg_marglik.backward()
            hyper_optimizer.step()

        # # Serialization for fitted quantities
        # state_dict = la.state_dict()
        # torch.save(state_dict, "state_dict.bin")
        #
        # la = Laplace(model, "regression", subset_of_weights="all", hessian_structure="full")
        # # Load serialized, fitted quantities
        # la.load_state_dict(torch.load("state_dict.bin"))

        X_uq = self._arr_to_tensor(X_uq)
        f_mu, f_var = la(X_uq)

        f_mu = f_mu.squeeze().detach().cpu().numpy()
        f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
        pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)

        y_pred, y_std = f_mu, pred_std
        y_quantiles = self.quantiles_gaussian(quantiles, y_pred, y_std)
        return y_pred, y_quantiles, y_std

    def native_quantile_regression(self, X_train, y_train, X_uq, quantiles):
        y_pred, y_quantiles = estimate_quantiles_qr(
            X_train, y_train, X_uq, alpha=quantiles
        )
        y_std = self.stds_from_quantiles(y_quantiles)
        return y_pred, y_quantiles, y_std

    # noinspection PyMethodMayBeStatic
    # todo: make static?
    def native_gp(self, X_train, y_train, X_uq, quantiles, verbose=True):
        if verbose:
            print(f"fitting GP kernel... [{time.strftime('%H:%M:%S')}]")
        kernel = self._get_kernel()
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, random_state=0, normalize_y=False, n_restarts_optimizer=10
        )
        gaussian_process.fit(X_train, y_train)
        if verbose:
            print(f"done. [{time.strftime('%H:%M:%S')}]")
            print("kernel:", gaussian_process.kernel_)
            print("GP predicting...")
        mean_prediction, std_prediction = gaussian_process.predict(
            X_uq, return_std=True
        )
        if verbose:
            print("done.")
        y_pred, y_std = mean_prediction, std_prediction
        y_quantiles = self.quantiles_gaussian(quantiles, y_pred, y_std)
        return y_pred, y_quantiles, y_std

    @classmethod
    def _df_to_tensor(cls, df: pd.DataFrame, dtype=float) -> torch.Tensor:
        return cls._arr_to_tensor(df.to_numpy(dtype=dtype))

    @staticmethod
    def _arr_to_tensor(arr) -> torch.Tensor:
        return torch.Tensor(arr).float()

    @classmethod
    def _get_train_loader(
        cls, X_train: torch.Tensor, y_train: torch.Tensor, batch_size
    ):
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        return train_loader

    @staticmethod
    def quantiles_gaussian(quantiles, y_pred, y_std):
        # todo: does this work for multi-dim outputs?
        return np.array(
            [
                norm.ppf(quantiles, loc=mean, scale=std)
                for mean, std in zip(y_pred, y_std)
            ]
        )

    @staticmethod
    def _get_kernel():
        return RBF() + WhiteKernel()


def main():
    uq_comparer = My_UQ_Comparer(
        method_whitelist=METHOD_WHITELIST, to_standardize=TO_STANDARDIZE
    )
    uq_metrics = uq_comparer.compare_methods(
        QUANTILES,
        should_plot_data=PLOT_DATA,
        should_plot_results=PLOT_RESULTS,
        should_save_plots=SAVE_PLOTS,
        plots_path=PLOTS_PATH,
        base_model_params=BASE_MODEL_PARAMS,
        output_uq_on_train=True,
        return_results=False,
    )
    print_metrics(uq_metrics)


if __name__ == "__main__":
    main()
