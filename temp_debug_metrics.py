print('importing')

from typing import Generator

import numpy as np

import settings
from helpers import misc_helpers
from helpers.io_helper import IO_Helper
print('done')


def main():
    print('running main')
    print('loading data')
    io_helper = IO_Helper(
        settings.STORAGE_PATH,
        methods_kwargs=settings.METHODS_KWARGS,
        filename_parts=settings.FILENAME_PARTS,
        n_samples=settings.N_POINTS_PER_GROUP,
    )
    y_pred = io_helper.load_array(filename='native_mvnn_y_pred_n800_it100_nh2_hs20.npy')
    y_quantiles = io_helper.load_array(filename='native_mvnn_y_quantiles_n800_it100_nh2_hs20.npy')
    y_std = io_helper.load_array(filename='native_mvnn_y_std_n800_it100_nh2_hs20.npy')

    X_train, X_test, y_train, y_test, X, y, scaler_y = misc_helpers.get_data(
        filepath=settings.DATA_FILEPATH,
        n_points_per_group=settings.N_POINTS_PER_GROUP,
        standardize_data=True,
    )
    y_true_orig_scale = misc_helpers.inverse_transform_y(scaler_y, y)

    metrics_det = compute_metrics_det(y_pred, y_true_orig_scale)
    print('metrics det:', metrics_det)
    metrics_uq = compute_metrics_uq(y_pred, y_quantiles, y_std, y_true_orig_scale, settings.QUANTILES)
    print('metrics uq:', metrics_uq)


def compute_metrics_det(y_pred, y_true) -> dict[str, float]:
    print('computing det metrics')

    from helpers.metrics import rmse, smape_scaled

    y_pred, y_true = _clean_ys_for_metrics(y_pred, y_true)
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "smape_scaled": smape_scaled(y_true, y_pred),
    }
    return _clean_metrics(metrics)


def compute_metrics_uq(y_pred, y_quantiles, y_std, y_true, quantiles) -> dict[str, float]:
    print('computing uq metrics')

    from helpers.metrics import crps, nll_gaussian, mean_pinball_loss

    y_pred, y_quantiles, y_std, y_true = _clean_ys_for_metrics(y_pred, y_quantiles, y_std, y_true)
    metrics = {
        "crps": crps(y_true, y_quantiles),
        "nll_gaussian": nll_gaussian(y_true, y_pred, y_std),
        "mean_pinball": mean_pinball_loss(y_pred, y_quantiles, quantiles),
    }
    return _clean_metrics(metrics)


def _clean_ys_for_metrics(*ys) -> Generator[np.ndarray | None, None, None]:
    for y in ys:
        if y is None:
            yield y
        else:
            yield np.array(y).squeeze()


def _clean_metrics(metrics):
    metrics = {
        metric_name: (None if value is None else float(value))
        for metric_name, value in metrics.items()
    }
    return metrics


if __name__ == '__main__':
    main()
