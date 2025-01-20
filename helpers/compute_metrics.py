import logging
from functools import partial
from typing import Generator, Union, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import numpy as np


def compute_metrics_det(y_pred, y_true, metrics_whitelist: set[str] = None) -> dict[str, float] | None:
    """

    :param metrics_whitelist: only compute these metrics. if None, compute all.
    :param y_pred:
    :param y_true:
    :return:
    """
    if metrics_whitelist is not None and not metrics_whitelist:
        logging.info('no deterministic metrics specified. skipping...')
        return

    from helpers._metrics import rmse, smape_scaled, mae

    y_pred, y_true = _make_arrs_1d_allow_none(y_pred, y_true)
    metrics_funcs = {
        "mae": partial(mae, y_true, y_pred),
        "rmse": partial(rmse, y_true, y_pred),
        "smape_scaled": partial(smape_scaled, y_true, y_pred),
    }
    metrics = _metrics_funcs_dict_to_metrics_dict(metrics_funcs, metrics_whitelist)
    return _metrics_to_float_allow_none(metrics)


def compute_metrics_uq(
        y_pred, y_quantiles, y_std, y_true, quantiles, coverage_level=0.9, metrics_whitelist: set[str] = None,
    ) -> dict[str, float] | None:
    """

    :param metrics_whitelist: only compute these metrics. if None, compute all.
    :param y_pred:
    :param y_quantiles:
    :param y_std:
    :param y_true:
    :param quantiles:
    :param coverage_level: assumed to be 2-digit float in [0.01, 0.99]
    :return:
    """
    if metrics_whitelist is not None and not metrics_whitelist:
        logging.info('no UQ metrics specified. skipping...')
        return

    from helpers._metrics import crps, nll_gaussian, mean_pinball_loss, ssr, coverage

    y_pred, y_quantiles, y_std, y_true = _make_arrs_1d_allow_none(y_pred, y_quantiles, y_std, y_true)
    if metrics_whitelist is None or 'coverage' in metrics_whitelist:
        logging.info(f'coverage will be computed at level {coverage_level}')
    metrics_funcs = {
        "crps": partial(crps, y_true, y_quantiles),
        "nll_gaussian": partial(nll_gaussian, y_true, y_pred, y_std),
        "mean_pinball": partial(mean_pinball_loss, y_pred, y_quantiles, quantiles),
        "ssr": partial(ssr, y_true, y_pred, y_std),
        "coverage": partial(coverage, y_true, y_quantiles, quantiles, coverage_level=coverage_level),
    }
    metrics = _metrics_funcs_dict_to_metrics_dict(metrics_funcs, metrics_whitelist)
    return _metrics_to_float_allow_none(metrics)


### HELPERS ###


def _metrics_funcs_dict_to_metrics_dict(metrics_funcs: dict[str, Callable], metrics_whitelist: set[str] | None):
    metrics = {
        metric_name: metric_func()
        for metric_name, metric_func in metrics_funcs.items()
        if metrics_whitelist is None or metric_name.lower() in metrics_whitelist
    }
    return metrics


def _metrics_to_float_allow_none(metrics: dict):
    """
    metrics are allowed to be None
    :param metrics:
    :return:
    """
    metrics = {
        metric_name: (None if value is None else float(value))
        for metric_name, value in metrics.items()
    }
    return metrics


def _make_arrs_1d_allow_none(*arrs) -> Generator[Union['np.ndarray', None], None, None]:
    """
    ys are allowed to be None
    :param metrics:
    :return:
    """
    import numpy as np
    for arr in arrs:
        if arr is None:
            yield arr
        else:
            yield np.array(arr).squeeze()
