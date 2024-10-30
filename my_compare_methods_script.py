from typing import Any

import numpy as np
# import numpy.typing as npt

import pandas as pd
from mapie.subsample import BlockBootstrap
from pmdarima.metrics import smape

from scipy.stats import randint, norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_pinball_loss
from statsmodels.tools.eval_measures import rmse
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from uncertainty_toolbox.metrics_scoring_rule import nll_gaussian
# from properscoring import crps_ensemble

from compare_methods import UQ_Comparer
from helpers import get_data, IO_Helper

from conformal_prediction import (
    train_base_model as train_base_model_cp,
    estimate_pred_interals_no_pfit_enbpi,
)
from quantile_regression import estimate_quantiles as estimate_quantiles_qr

import torch

from laplace import Laplace


METHOD_WHITELIST = [
    # "posthoc_conformal_prediction",
    # "posthoc_laplace",
    # "native_quantile_regression",
    "native_gp",
]
QUANTILES = [0.05, 0.25, 0.75, 0.95]  # todo: how to handle 0.5? ==> just use mean if needed

PLOT_DATA = False
PLOT_RESULTS = True  # todo: fix plotting timing?
SAVE_PLOTS = True

PLOTS_PATH = "plots"

BASE_MODEL_PARAMS = {
    'skip_training': True,
    # 'n_jobs': -1,
    # 'model_params_choices': None,
}

torch.set_default_dtype(torch.float32)


# noinspection PyPep8Naming
class My_UQ_Comparer(UQ_Comparer):
    def __init__(self, storage_path="comparison_storage", *args, **kwargs):
        """

        :param storage_path:
        :param args: passed to super.__init__
        :param kwargs: passed to super.__init__
        """
        super().__init__(*args, **kwargs)
        self.io_helper = IO_Helper(storage_path)

    # todo: remove param?
    def get_data(self, _n_points_per_group=100):
        return get_data(_n_points_per_group, return_full_data=True)

    # todo: type hints!
    def compute_metrics(self, y_pred, y_quantiles, y_std, y_true, quantiles=None):
        """

        :param y_pred: predicted y-values
        :param y_quantiles:
        :param y_std:
        :param y_true:
        :param quantiles:
        :return:
        """
        y_true_np = y_true.to_numpy().squeeze()
        # todo: sharpness? calibration? PIT? coverage?
        # todo: skill score (but what to use as benchmark)?

        metrics = {  # todo: improve
            "rmse": rmse(y_true_np, y_pred),
            "smape": smape(y_true_np, y_pred) / 100,  # scale down to [0, 1]
            "crps": (
                # todo: implement
                None  # crps_ensemble(y_pred, y_std, y_true_np) if y_std is not None else None
            ),
            "neg_log_lik": (
                nll_gaussian(y_pred, y_std, y_true_np) if y_std is not None else None
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
        return np.mean(
            [
                mean_pinball_loss(y_true, y_quantiles[:, ind], alpha=quantile)
                for ind, quantile in enumerate(quantiles)
            ]
        )

    def train_base_model(
        self, X_train, y_train, model_params_choices=None, skip_training=True, n_jobs=-1
    ):
        # todo: more flexibility in choosing (multiple) base models
        if model_params_choices is None:
            model_params_choices = {
                "max_depth": randint(2, 30),
                "n_estimators": randint(10, 100),
            }
        return train_base_model_cp(
            RandomForestRegressor,
            model_params_choices=model_params_choices,
            X_train=X_train,
            y_train=y_train,
            skip_training=skip_training,
            cv_n_iter=10,
            n_jobs=n_jobs,
            io_helper=self.io_helper,
        )

    def train_base_model2(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        model_params_choices=None,
        n_epochs=1000,
        batch_size=1,
        random_state=711,
        verbose=True,
        skip_training=True,
        save_trained=True,
        model_filename="_laplace_base.pth",
    ):
        """

        :param model_filename:
        :param save_trained:
        :param skip_training:
        :param verbose:
        :param X_train: shape (n_samples, n_dims)
        :param y_train: shape (n_samples, n_dims)
        :param model_params_choices:
        :param n_epochs:
        :param batch_size:
        :param random_state:
        :return:
        """
        # todo: more flexibility in choosing (multiple) base models
        torch.manual_seed(random_state)

        dim_in, dim_out = X_train.shape[-1], y_train.shape[-1]
        model = torch.nn.Sequential(
            torch.nn.Linear(dim_in, 50), torch.nn.Tanh(), torch.nn.Linear(50, dim_out)
        ).float()

        if skip_training:
            print("skipping base model training")
            try:
                model = self.io_helper.load_torch_model(
                    model_filename, weights_only=False
                )
                model.eval()
                return model
            except FileNotFoundError:
                # todo
                print(
                    "error. model not found, so training cannot be skipped. training from scratch"
                )

        # todo: consistent input expectations!
        train_loader = self._get_train_loader(X_train, y_train, batch_size)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        iterable = tqdm(range(n_epochs)) if verbose else range(n_epochs)
        for _ in iterable:
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
        if save_trained:
            torch.save(model, model_filename)
        model.eval()
        return model

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
            skip_base_training=True,
            io_helper=self.io_helper,
        )
        # todo!
        y_quantiles = self.quantiles_from_pis(y_pis)
        y_std = None  # self.stds_from_quantiles(y_quantiles)
        return y_pred, y_quantiles, y_std

    def posthoc_laplace(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_uq: pd.DataFrame,
        quantiles,
        model,
        n_epochs=1000,
        batch_size=1,
        random_state=711,
        verbose=True,
    ):
        # todo: offer option to alternatively optimize parameters and hyperparameters of the prior jointly (cf. example
        #  script)?
        train_loader = self._get_train_loader(X_train, y_train, batch_size)

        la = Laplace(model, "regression")
        la.fit(train_loader)
        log_prior, log_sigma = (
            torch.ones(1, requires_grad=True),
            torch.ones(1, requires_grad=True),
        )
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
        iterable = tqdm(range(n_epochs)) if verbose else range(n_epochs)
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

        X_uq = self._df_to_tensor(X_uq)
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
    def native_gp(self, X_train, y_train, X_uq, quantiles):
        kernel = self._get_kernel()
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=200
        )
        gaussian_process.fit(X_train, y_train)

        mean_prediction, std_prediction = gaussian_process.predict(
            X_uq, return_std=True
        )
        y_pred, y_std = mean_prediction, std_prediction
        y_quantiles = self.quantiles_gaussian(quantiles, y_pred, y_std)
        return y_pred, y_quantiles, y_std

    @staticmethod
    def _df_to_tensor(df: pd.DataFrame, dtype=float) -> torch.Tensor:
        return torch.Tensor(df.to_numpy(dtype=dtype))

    @classmethod
    def _get_train_loader(cls, X_train, y_train, batch_size):
        X_train, y_train = map(
            lambda df: cls._df_to_tensor(df, dtype=float), (X_train, y_train)
        )
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
        # old kernel: 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

        # todo: how to set values and bounds without "cheating"
        output_shift, output_shift_bounds = 1, (1e-6, 1e6)
        output_scale, output_scale_bounds = 1, (1e-6, 1e6)
        rbf_scale, rbf_scale_bounds = 1, (1e-6, 1e6)
        noise_scale, noise_scale_bounds = 1, (1e-6, 1e6)

        ## better values:
        # output_shift, output_shift_bounds = 13_000, (10_000, 20_000)
        # output_scale, output_scale_bounds = 5000, (5000, 40000)
        # rbf_scale, rbf_scale_bounds = 100, (10, 500)
        # noise_scale, noise_scale_bounds = 100, (0.1, 1000)
        
        def kernelize(kernel, value, bounds):
            return kernel(value**2, (bounds[0] ** 2, bounds[1] ** 2))

        # fmt: off
        # elaborating values explicitly
        kernel = (
            kernelize(ConstantKernel, output_shift, output_shift_bounds)
            + (kernelize(ConstantKernel, output_scale, output_scale_bounds)
               * kernelize(RBF, rbf_scale, rbf_scale_bounds))
            + kernelize(WhiteKernel, noise_scale, noise_scale_bounds)
        )
        return kernel


def print_metrics(uq_metrics: dict[str, dict[str, dict[str, Any]]]):
    print()
    for uq_type, method_metrics in uq_metrics.items():
        print(f'{uq_type} metrics:')
        for method, metrics in method_metrics.items():
            print(f'\t{method}:')
            for metric, value in metrics.items():
                print(f'\t\t{metric}: {value}')
        print()


def main():
    uq_comparer = My_UQ_Comparer(method_whitelist=METHOD_WHITELIST)
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
