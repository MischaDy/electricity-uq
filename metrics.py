import numpy as np

from uncertainty_toolbox.metrics_scoring_rule import nll_gaussian as nll_gaussian_
from sklearn.metrics import mean_pinball_loss as mean_pinball_loss_
# from scores.probability import crps_cdf
# import xarray


def smape_scaled(y_true, y_pred):
    """
    output 100 * smape, which is in [0, 2] (?)
    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: why such terrible perf?
    numerators = np.abs(y_pred - y_true)
    denominators = (np.abs(y_pred) + np.abs(y_true)) / 2
    return np.mean(numerators/denominators)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def mean_pinball_loss(y_true, y_quantiles, quantiles):
    """

    :param y_true:
    :param y_quantiles: array of shape (n_samples, n_quantiles)
    :param quantiles:
    :return:
    """
    if y_quantiles is None:
        return None
    # fmt: off
    return np.mean([
        mean_pinball_loss_(y_true, y_quantiles[:, ind], alpha=quantile)
        for ind, quantile in enumerate(quantiles)
    ])


def nll_gaussian(y_true, y_pred, y_std):
    if y_std is None:
        return None
    return nll_gaussian_(y_pred, y_std, y_true)


def crps(y_true, y_pred, y_std):
    # todo: implement
    # https://scores.readthedocs.io/en/stable/tutorials/CRPS_for_CDFs.html
    # crps_ensemble(y_pred, y_std, y_true_np) if y_std is not None else None
    # crps_for_ensemble(ensemble_forecast, obs_array, ensemble_member_dim="ensemble_member", method="fair")
    # crps_cdf
    return None
