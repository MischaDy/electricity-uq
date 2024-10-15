import numpy as np
import pandas as pd
from mapie.subsample import BlockBootstrap

from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_pinball_loss
from torch.utils.data import TensorDataset, DataLoader
from uncertainty_toolbox.metrics_scoring_rule import crps_gaussian, nll_gaussian

from compare_methods import UQ_Comparer
from helpers import get_data

from conformal_prediction import (
    train_base_model,
    estimate_pred_interals_no_pfit_enbpi,
)
from quantile_regression import estimate_quantiles as estimate_quantiles_qr

import torch

from laplace import Laplace


QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
PLOT_DATA = False
PLOT_RESULTS = False


# noinspection PyPep8Naming
class My_UQ_Comparer(UQ_Comparer):
    # todo: remove param?
    def get_data(self, _n_points_per_group=100):
        return get_data(_n_points_per_group, return_full_data=True)

    def compute_metrics(self, y_pred, y_quantiles, y_std, y_true, quantiles=None):
        y_true_np = y_true.to_numpy().squeeze()
        metrics = {  # todo: improve
            "crps": crps_gaussian(y_pred, y_std, y_true_np) if y_std is not None else None,
            "neg_log_lik": nll_gaussian(y_pred, y_std, y_true_np) if y_std is not None else None,
            "mean_pinball": self._mean_pinball_loss(y_pred, y_quantiles, quantiles) if y_quantiles is not None else None,
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
        return np.mean([mean_pinball_loss(y_true, y_quantiles[:, ind], alpha=quantile)
                        for ind, quantile in enumerate(quantiles)])

    def train_base_model_normal(self, X_train, y_train, model_params_choices=None):
        # todo: more flexibility in choosing (multiple) base models
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

    def train_base_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        model_params_choices=None,
        n_epochs=100,
        batch_size=1,
        random_state=711,
    ):
        # todo: more flexibility in choosing (multiple) base models
        torch.manual_seed(random_state)

        model = torch.nn.Sequential(
            torch.nn.Linear(1, 50), torch.nn.Tanh(), torch.nn.Linear(50, 1)
        )

        # todo: consistent input expectations!
        X_train, y_train = map(self._df_to_tensor, (X_train, y_train))
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for i in range(n_epochs):
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
        return model

    def posthoc_conformal_prediction(
        self, X_train, y_train, X_test, quantiles, model, random_state=42
    ):
        cv = BlockBootstrap(
            n_resamplings=10, n_blocks=10, overlapping=False, random_state=random_state
        )
        alphas = self.pis_from_quantiles(quantiles)
        y_pred, y_pis = estimate_pred_interals_no_pfit_enbpi(
            model,
            cv,
            alphas,
            X_test,
            X_train,
            y_train,
            skip_base_training=True,
        )
        y_std = self.stds_from_quantiles(y_pis)
        y_quantiles = None  # todo: sth with y_pis
        return y_pred, y_quantiles, y_std

    def posthoc_laplace(
        self,
        X_train,
        y_train,
        X_test,
        quantiles,
        model,
        n_epochs=100,
        batch_size=1,
        random_state=711,
    ):
        # todo: offer option to alternatively optimize parameters and hyperparameters of the prior jointly (cf. example
        #  script)?
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        la = Laplace(
            model, "regression"
        )  # , subset_of_weights="all", hessian_structure="full")
        la.fit(train_loader)
        log_prior, log_sigma = (
            torch.ones(1, requires_grad=True),
            torch.ones(1, requires_grad=True),
        )
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
        for i in range(n_epochs):
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

        # x = X_test.flatten().cpu().numpy()

        # Two options:
        # 1.) Marginal predictive distribution N(f_map(x_i), var(x_i))
        # The mean is (m,k), the var is (m,k,k)
        f_mu, f_var = la(X_test)

        # # 2.) Joint pred. dist. N((f_map(x_1),...,f_map(x_m)), Cov(f(x_1),...,f(x_m)))
        # # The mean is (m*k,) where k is the output dim. The cov is (m*k,m*k)
        # f_mu_joint, f_cov = la(X_test, joint=True)

        # # Both should be true
        # assert torch.allclose(f_mu.flatten(), f_mu_joint)
        # assert torch.allclose(f_var.flatten(), f_cov.diag())

        f_mu = f_mu.squeeze().detach().cpu().numpy()
        f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
        pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)

        # plot_regression(
        #     X_train, y_train, x, f_mu, pred_std, file_name="regression_example", plot=True, file_path='.'
        # )

        y_pred, y_quantiles, y_std = f_mu, None, pred_std

        # model = get_model()
        # la, model, margliks, losses = marglik_training(
        #     model=model,
        #     train_loader=train_loader,
        #     likelihood="regression",
        #     hessian_structure="full",
        #     backend=BackPackGGN,
        #     n_epochs=n_epochs,
        #     optimizer_kwargs={"lr": 1e-2},
        #     prior_structure="scalar",
        # )
        #
        # print(
        #     f"sigma={la.sigma_noise.item():.2f}",
        #     f"prior precision={la.prior_precision.numpy()}",
        # )
        #
        # f_mu, f_var = la(X_test)
        # f_mu = f_mu.squeeze().detach().cpu().numpy()
        # f_sigma = f_var.squeeze().sqrt().cpu().numpy()
        # pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)
        # plot_regression(
        #     X_train,
        #     y_train,
        #     x,
        #     f_mu,
        #     pred_std,
        #     file_name="regression_example_online",
        #     plot=False,
        # )
        return y_pred, y_quantiles, y_std

    def native_quantile_regression(self, X_train, y_train, X_test, quantiles):
        y_pred, y_quantiles = estimate_quantiles_qr(X_train, y_train, X_test, alpha=quantiles)
        y_std = self.stds_from_quantiles(y_quantiles)
        return y_pred, y_quantiles, y_std

    @staticmethod
    def _df_to_tensor(df: pd.DataFrame) -> torch.Tensor:
        return torch.Tensor(df.to_numpy())


def main():
    uq_comparer = My_UQ_Comparer()
    uq_comparer.compare_methods(QUANTILES, should_plot_data=PLOT_DATA, should_plot_results=PLOT_RESULTS)


if __name__ == "__main__":
    main()
