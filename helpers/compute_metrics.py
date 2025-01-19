from typing import Generator, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def compute_metrics_det(y_pred, y_true) -> dict[str, float]:
    from helpers._metrics import rmse, smape_scaled, mae

    y_pred, y_true = _make_arrs_1d_allow_none(y_pred, y_true)
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape_scaled": smape_scaled(y_true, y_pred),
    }
    return _metrics_to_float_allow_none(metrics)


def compute_metrics_uq(y_pred, y_quantiles, y_std, y_true, quantiles) -> dict[str, float]:
    from helpers._metrics import crps, nll_gaussian, mean_pinball_loss, ssr

    y_pred, y_quantiles, y_std, y_true = _make_arrs_1d_allow_none(y_pred, y_quantiles, y_std, y_true)
    metrics = {
        "crps": crps(y_true, y_quantiles),
        "nll_gaussian": nll_gaussian(y_true, y_pred, y_std),
        "mean_pinball": mean_pinball_loss(y_pred, y_quantiles, quantiles),
        "ssr": ssr(y_true, y_pred, y_std),
    }
    return _metrics_to_float_allow_none(metrics)


### HELPERS ###

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
