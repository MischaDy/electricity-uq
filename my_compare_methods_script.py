from pprint import pprint

import numpy as np
import numpy.typing as npt

import pandas as pd
from mapie.subsample import BlockBootstrap

from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_pinball_loss
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from uncertainty_toolbox.metrics_scoring_rule import crps_gaussian, nll_gaussian

from compare_methods import UQ_Comparer
from helpers import get_data

from conformal_prediction import (
    train_base_model as train_base_model_cp,
    estimate_pred_interals_no_pfit_enbpi,
)
from quantile_regression import estimate_quantiles as estimate_quantiles_qr

import torch

from laplace import Laplace


QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
PLOT_DATA = True
PLOT_RESULTS = True
SAVE_PLOTS = True


torch.set_default_dtype(torch.float32)


# noinspection PyPep8Naming
class My_UQ_Comparer(UQ_Comparer):
    # todo: remove param?
    def get_data(self, _n_points_per_group=100):
        return get_data(_n_points_per_group, return_full_data=True)

    def compute_metrics(self, y_pred, y_quantiles, y_std, y_true, quantiles=None):
        y_true_np = y_true.to_numpy().squeeze()
        metrics = {  # todo: improve
            "crps": (
                crps_gaussian(y_pred, y_std, y_true_np) if y_std is not None else None
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

    def train_base_model_normal(self, X_train, y_train, model_params_choices=None):
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
            load_trained_model=True,
            cv_n_iter=10,
        )

    def train_base_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        model_params_choices=None,
        n_epochs=100,
        batch_size=1,
        random_state=711,
        verbose=True,
        load_trained=True,
        save_trained=True,
        model_path="_laplace_base.pth",
    ):
        """

        :param model_path:
        :param save_trained:
        :param load_trained:
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

        if load_trained:
            print("skipping base model training")
            try:
                model = torch.load(model_path, weights_only=False)
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
            torch.save(model, model_path)
        model.eval()
        return model

    #
    # def posthoc_conformal_prediction(
    #     self, X_train, y_train, X_test, quantiles, model, random_state=42
    # ):
    #     cv = BlockBootstrap(
    #         n_resamplings=10, n_blocks=10, overlapping=False, random_state=random_state
    #     )
    #     alphas = self.pis_from_quantiles(quantiles)
    #     y_pred, y_pis = estimate_pred_interals_no_pfit_enbpi(
    #         model,
    #         cv,
    #         alphas,
    #         X_test,
    #         X_train,
    #         y_train,
    #         skip_base_training=True,
    #     )
    #     y_std = self.stds_from_quantiles(y_pis)
    #     y_quantiles = None  # todo: sth with y_pis
    #     return y_pred, y_quantiles, y_std

    def posthoc_laplace(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        quantiles,
        model,
        n_epochs=100,
        batch_size=1,
        random_state=711,
        verbose=True,
    ):
        # todo: offer option to alternatively optimize parameters and hyperparameters of the prior jointly (cf. example
        #  script)?
        train_loader = self._get_train_loader(X_train, y_train, batch_size)

        la = Laplace(
            model, "regression"
        )  # , subset_of_weights="all", hessian_structure="full")
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

        X_test = self._df_to_tensor(X_test)
        f_mu, f_var = la(X_test)

        f_mu = f_mu.squeeze().detach().cpu().numpy()
        f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
        pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)

        y_pred, y_quantiles, y_std = f_mu, None, pred_std
        return y_pred, y_quantiles, y_std

    def native_quantile_regression(self, X_train, y_train, X_test, quantiles):
        y_pred, y_quantiles = estimate_quantiles_qr(
            X_train, y_train, X_test, alpha=quantiles
        )
        y_std = self.stds_from_quantiles(y_quantiles)
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


def print_metrics(native_metrics, posthoc_metrics):
    pprint(native_metrics)
    pprint(posthoc_metrics)


def main():
    uq_comparer = My_UQ_Comparer()
    native_metrics, posthoc_metrics = uq_comparer.compare_methods(
        QUANTILES,
        should_plot_data=PLOT_DATA,
        should_plot_results=PLOT_RESULTS,
        should_save_plots=SAVE_PLOTS,
    )
    print_metrics(native_metrics, posthoc_metrics)


if __name__ == "__main__":
    main()
