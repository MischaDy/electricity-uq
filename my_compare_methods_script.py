import numpy as np
from mapie.subsample import BlockBootstrap

from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_pinball_loss
from uncertainty_toolbox.metrics_scoring_rule import crps_gaussian, nll_gaussian

from compare_methods import UQ_Comparer
from helpers import get_data

from conformal_prediction import (
    train_base_model,
    estimate_pred_interals_no_pfit_enbpi,
)
from quantile_regression import estimate_quantiles as estimate_quantiles_qr


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

    def train_base_model(self, X_train, y_train, model_params_choices=None):
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

    def posthoc_conformal_prediction(
        self, X_train, y_train, X_test, quantiles, model, random_state=42
    ):
        cv_mapie_ts = BlockBootstrap(
            n_resamplings=10, n_blocks=10, overlapping=False, random_state=random_state
        )
        alphas = ...  # todo: sth with quantiles
        y_pred, y_pis = estimate_pred_interals_no_pfit_enbpi(
            model,
            cv_mapie_ts,
            alphas,
            X_test,
            X_train,
            y_train,
            skip_base_training=True,
        )
        # shape y_pis: (n_samples, 2, n_alphas)
        # todo: reasonable with few quantiles?
        y_std = self.stds_from_quantiles(y_pis)
        return y_pred, y_pis, y_std

    def native_quantile_regression(self, X_train, y_train, X_test, quantiles):
        y_pred, y_quantiles = estimate_quantiles_qr(X_train, y_train, X_test, alpha=quantiles)
        # todo: reasonable with few quantiles?
        y_std = self.stds_from_quantiles(y_quantiles)
        return y_pred, y_quantiles, y_std


def main():
    uq_comparer = My_UQ_Comparer()
    uq_comparer.compare_methods(QUANTILES, should_plot_data=PLOT_DATA, should_plot_results=PLOT_RESULTS)


if __name__ == "__main__":
    main()
