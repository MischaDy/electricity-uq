import numpy as np

from uncertainty_toolbox.metrics_scoring_rule import nll_gaussian as nll_gaussian_
from sklearn.metrics import mean_pinball_loss as mean_pinball_loss_
import scores
import xarray


def smape_scaled(y_true: np.ndarray, y_pred: np.ndarray):
    """
    output 100 * smape, which is in [0, 2]
    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: why such terrible perf?
    numerators = np.abs(y_pred - y_true)
    denominators = (np.abs(y_pred) + np.abs(y_true)) / 2
    return np.mean(numerators / denominators)


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(mse(y_true, y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean((y_pred - y_true) ** 2)


def mean_pinball_loss(y_true: np.ndarray, y_quantiles: np.ndarray, quantiles: np.ndarray):
    """

    :param y_true:
    :param y_quantiles: array of shape (n_samples, n_quantiles)
    :param quantiles:
    :return:
    """
    if y_quantiles is None:
        return None
    return np.mean([
        mean_pinball_loss_(y_true, y_quantiles[:, ind], alpha=quantile)
        for ind, quantile in enumerate(quantiles)
    ])


def nll_gaussian(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray):
    if y_std is None:
        return None
    return nll_gaussian_(y_pred, y_std, y_true)


def crps(y_true: np.ndarray, y_quantiles: np.ndarray):
    """
    We interpret y_quantiles as a "deterministic sample" from the underlying predicted distribution. Hence, we use
    method='fair'. See the docs for details:
    https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_for_ensemble

    :param y_true:
    :param y_quantiles:
    :return:
    """
    if y_quantiles is None:
        return None
    sample_dim, ensemble_member_dim = 'samples', 'sample_quantiles'
    y_true = xarray.DataArray(y_true, dims=[sample_dim])
    y_quantiles = xarray.DataArray(y_quantiles, dims=[sample_dim, ensemble_member_dim])
    crps = scores.probability.crps_for_ensemble(
        y_quantiles, y_true, ensemble_member_dim=ensemble_member_dim, method='fair'
    )
    return crps.to_numpy()
