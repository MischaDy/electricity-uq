"""
all metrics expect 1D arrays
all UQ metrics allow y_std and/or y_quantiles to be None, in which case they return None
"""

import numpy as np


### DETERMINISTIC ###

def smape_scaled(y_true: np.ndarray, y_pred: np.ndarray):
    """
    output 100 * smape, which is in [0, 2]
    :param y_true:
    :param y_pred:
    :return:
    """
    numerators = np.abs(y_pred - y_true)
    denominators = (np.abs(y_pred) + np.abs(y_true)) / 2
    return np.mean(numerators / denominators)


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(mse(y_true, y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean((y_pred - y_true) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray, keep_dim=False):
    """

    :param y_true:
    :param y_pred:
    :param keep_dim:
    :return: the MAE if keep_dim False, else just the absolute errors array
    """
    absolute_errors = np.abs(y_true - y_pred)
    return absolute_errors if keep_dim else np.mean(absolute_errors)


### PROBABILISTIC ##


def coverage(y_true: np.ndarray, y_quantiles: np.ndarray, quantiles: list[float], coverage_level: float,
             keep_dim=False, verbose=True):
    quantile_ind_low, quantile_ind_high = _get_quantile_inds(quantiles, pi_interval=coverage_level)
    pred_below_data = (y_quantiles[:, quantile_ind_low] <= y_true)
    pred_above_data = (y_true <= y_quantiles[:, quantile_ind_high])
    if verbose:
        print(f'{coverage_level=}: avg. pred below {pred_below_data.mean()}; avg. pred above {pred_above_data.mean()}')
    coverage_arr = pred_below_data & pred_above_data
    return coverage_arr if keep_dim else coverage_arr.mean()


def ssr(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, keep_dim=False, eps=1e-9):
    """
    compute spread-skill ratio

    :param eps: lower bound of denominator to prevent division by (almost) zero
    :param y_std:
    :param y_true:
    :param y_pred:
    :param keep_dim:
    :return: std/AE if keep_dim, else std/RMSE
    """
    if y_std is None:
        return None

    if keep_dim:
        spread = y_std
        skill = mae(y_true, y_pred, keep_dim=True)  # RMSE without mean is just AEs
    else:
        spread = np.mean(y_std)
        skill = rmse(y_true, y_pred)

    assert eps >= 0
    eps_arr = np.zeros_like(skill) + eps
    skill = np.max([skill, eps_arr], axis=0)
    return spread / skill


def mean_pinball_loss(y_true: np.ndarray, y_quantiles: np.ndarray, quantiles: np.ndarray, keep_dim=False):
    """

    :param keep_dim:
    :param y_true:
    :param y_quantiles: array of shape (n_samples, n_quantiles)
    :param quantiles:
    :return:
    """
    from sklearn.metrics import mean_pinball_loss as mean_pinball_loss_
    if y_quantiles is None:
        return None
    loss = [
        mean_pinball_loss_(y_true, y_quantiles[:, ind], alpha=quantile)
        for ind, quantile in enumerate(quantiles)
    ]
    return loss if keep_dim else np.mean(loss)


def nll_gaussian(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, keep_dim=False):
    # based on uncertainty_toolbox.metrics_scoring_rule.nll_gaussian
    if y_std is None:
        return None

    from scipy import stats

    residuals = y_pred - y_true
    nll = -1 * stats.norm.logpdf(residuals, scale=y_std)
    return nll if keep_dim else np.mean(nll)


def crps(y_true: np.ndarray, y_quantiles: np.ndarray, keep_dim=False):
    """
    We interpret y_quantiles as a "deterministic sample" from the underlying predicted distribution. Hence, we use
    method='fair'. See the docs for details:
    https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_for_ensemble

    :param keep_dim:
    :param y_true:
    :param y_quantiles:
    :return:
    """
    import scores
    import xarray
    if y_quantiles is None:
        return None
    sample_dim, ensemble_member_dim = 'samples', 'sample_quantiles'
    y_true = xarray.DataArray(y_true, dims=[sample_dim])
    y_quantiles = xarray.DataArray(y_quantiles, dims=[sample_dim, ensemble_member_dim])

    preserve_dims = [sample_dim] if keep_dim else None
    crps = scores.probability.crps_for_ensemble(
        y_quantiles, y_true, ensemble_member_dim=ensemble_member_dim, method='fair', preserve_dims=preserve_dims,
    )
    return crps.to_numpy()


### HELPERS


def _get_quantile_inds(quantiles, pi_interval=0.9):
    assert 0 < pi_interval < 1
    quantiles = np.array(quantiles)
    shift = (1 - pi_interval) / 2
    quantile_low, quantile_high = shift, pi_interval + shift
    diffs_low, diffs_high = np.abs(quantiles - quantile_low), np.abs(quantiles - quantile_high)
    quantile_ind_low, quantile_ind_high = np.argmin(diffs_low), np.argmin(diffs_high)
    return quantile_ind_low, quantile_ind_high
