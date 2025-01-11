import logging

from typing import Generator, Union, TYPE_CHECKING

import settings
import settings_update
from helpers import misc_helpers
from helpers.io_helper import IO_Helper
from helpers.uq_arr_helpers import get_uq_method_to_arrs_dict

if TYPE_CHECKING:
    import numpy as np

logging.basicConfig(level=logging.INFO)


RUN_SIZE = 'full'
METHODS = {
    'qhgbr',
}
UQ_METHOD_TO_ARR_NAMES_DICT = {
    'qhgbr': [
        'native_qhgbr_y_pred_n210432_it0.npy',
        'native_qhgbr_y_quantiles_n210432_it0.npy',
        'native_qhgbr_y_std_n210432_it0.npy',
    ] if RUN_SIZE == 'full' else [
        'native_qhgbr_y_pred_n35136_it0.npy',
        'native_qhgbr_y_quantiles_n35136_it0.npy',
        'native_qhgbr_y_std_n35136_it0.npy',
    ]
}


def main():
    # todo: sharpness? calibration? PIT? coverage?
    logging.info('running metrics computation script')
    io_helper = IO_Helper()
    logging.info('loading train/test data')
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = _load_data()
    logging.info('loading predictions')
    uq_method_to_arrs_dict = get_uq_method_to_arrs_dict(uq_methods_whitelist=METHODS,
                                                        uq_method_to_arr_names_dict=UQ_METHOD_TO_ARR_NAMES_DICT)
    for method, arrs in uq_method_to_arrs_dict.items():
        logging.info(f'computing metrics for {method}:')
        y_pred, y_quantiles, y_std = arrs
        metrics = {}
        logging.info(f'deterministic metrics...')
        metrics_det = compute_metrics_det(y_pred, y)
        metrics.update(metrics_det)
        logging.info(f'UQ metrics...')
        metrics_uq = compute_metrics_uq(y_pred, y_quantiles, y_std, y, settings.QUANTILES)
        metrics.update(metrics_uq)
        logging.info(f'metrics: {metrics}')
        logging.info('saving metrics...')
        filename = f'uq_metrics_{method}'
        filename = misc_helpers.timestamped_filename(filename)
        io_helper.save_metrics(metrics, filename=filename)


def compute_metrics_det(y_pred, y_true) -> dict[str, float]:
    from helpers._metrics import rmse, smape_scaled

    y_pred, y_true = _make_arrs_1d_allow_none(y_pred, y_true)
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "smape_scaled": smape_scaled(y_true, y_pred),
    }
    return _metrics_to_float_allow_none(metrics)


def compute_metrics_uq(y_pred, y_quantiles, y_std, y_true, quantiles) -> dict[str, float]:
    from helpers._metrics import crps, nll_gaussian, mean_pinball_loss

    y_pred, y_quantiles, y_std, y_true = _make_arrs_1d_allow_none(y_pred, y_quantiles, y_std, y_true)
    metrics = {
        "crps": crps(y_true, y_quantiles),
        "nll_gaussian": nll_gaussian(y_true, y_pred, y_std),
        "mean_pinball": mean_pinball_loss(y_pred, y_quantiles, quantiles),
    }
    return _metrics_to_float_allow_none(metrics)


def _load_data():
    settings.RUN_SIZE = RUN_SIZE
    settings_update.update_run_size_setup()
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers.get_data(
        filepath=settings.DATA_FILEPATH,
        train_years=settings.TRAIN_YEARS,
        val_years=settings.VAL_YEARS,
        test_years=settings.TEST_YEARS,
        n_points_per_group=settings.N_POINTS_PER_GROUP,
        do_standardize_data=True,
    )
    y_train, y_val, y_test, y = map(scaler_y.inverse_transform, [y_train, y_val, y_test, y])
    return X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y


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


if __name__ == '__main__':
    main()
