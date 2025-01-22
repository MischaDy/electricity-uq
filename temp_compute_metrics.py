import logging

import settings
import settings_update
from helpers import misc_helpers
from helpers.compute_metrics import compute_metrics_uq, compute_metrics_det
from helpers.io_helper import IO_Helper
from helpers.arr_helpers import get_method_to_arrs_gen

logging.basicConfig(level=logging.INFO)

RUN_SIZE = 'full'
SHORTEN_TO_TEST = True
ARRAYS_FOLDER = 'arrays'
MODELS_FOLDER = 'models'
TIMESTAMPED_FILES = False

METRICS_WHITELIST_DET = set([
    # "mae",
    # "rmse",
    # "smape_scaled",
])
METRICS_WHITELIST_UQ = set([
    # "crps",
    # "nll_gaussian",
    "mean_pinball",
    # "ssr",
    # "coverage",
])

METHODS_WHITELIST = set([
    # 'base_model_hgbr',
    # 'base_model_linreg',
    # 'base_model_nn',
    # 'native_qhgbr',
    # 'native_qr',
    # 'native_gp',
    # 'native_mvnn',
    'posthoc_cp_hgbr',
    # 'posthoc_cp_linreg',
    # 'posthoc_cp_nn',
    # 'posthoc_la_nn',
])
UQ_METHOD_TO_ARR_NAMES_DICT = {
    'base_model_hgbr': ['base_model_hgbr_n210432_it30_its3.npy'],
    'base_model_linreg': ['base_model_linreg_n210432.npy'],
    'base_model_nn': ['base_model_nn_n210432_it400_nh2_hs50.npy'],
    'native_qhgbr': [
        'native_qhgbr_y_pred_n210432_it0.npy',
        'native_qhgbr_y_quantiles_n210432_it0.npy',
        'native_qhgbr_y_std_n210432_it0.npy',
    ],
    'native_qr': [
        'native_quantile_regression_nn_y_pred_n210432_it300_nh2_hs50.npy',
        'native_quantile_regression_nn_y_quantiles_n210432_it300_nh2_hs50.npy',
        'native_quantile_regression_nn_y_std_n210432_it300_nh2_hs50.npy',
    ],
    'native_gp': [
        'native_gpytorch_y_pred_n210432_it200_new.npy',
        'native_gpytorch_y_quantiles_n210432_it200_new.npy',
        'native_gpytorch_y_std_n210432_it200_new.npy',
    ],
    'native_mvnn': [
        'native_mvnn_y_pred_n210432_it100_nh2_hs50.npy',
        'native_mvnn_y_quantiles_n210432_it100_nh2_hs50.npy',
        'native_mvnn_y_std_n210432_it100_nh2_hs50.npy',
    ],
    'posthoc_cp_hgbr': [
        'posthoc_conformal_prediction_base_model_hgbr_y_pred_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_hgbr_y_quantiles_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_hgbr_y_std_n210432_it5.npy',
    ],
    'posthoc_cp_linreg': [
        'posthoc_conformal_prediction_base_model_linreg_y_pred_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_linreg_y_quantiles_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_linreg_y_std_n210432_it5.npy',
    ],
    'posthoc_cp_nn': [
        'posthoc_conformal_prediction_base_model_nn_y_pred_n210432_it5_cp2.npy',
        'posthoc_conformal_prediction_base_model_nn_y_quantiles_n210432_it5_cp2.npy',
        'posthoc_conformal_prediction_base_model_nn_y_std_n210432_it5_cp2.npy',
    ],
    'posthoc_la_nn': [
        'posthoc_laplace_approximation_base_model_nn_y_pred_n210432_it1000_la2.npy',
        'posthoc_laplace_approximation_base_model_nn_y_quantiles_n210432_it1000_la2.npy',
        'posthoc_laplace_approximation_base_model_nn_y_std_n210432_it1000_la2.npy',
    ],
}


def main():
    # todo: sharpness? calibration? PIT? coverage?
    logging.info('running metrics computation script')
    io_helper = IO_Helper(arrays_folder=ARRAYS_FOLDER, models_folder=MODELS_FOLDER)
    logging.info('loading train/test data')
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = _load_data()
    if SHORTEN_TO_TEST:
        logging.info('computing metrics on test data only')
        y_true = y_test
    else:
        y_true = y
    n_test_samples = y_test.shape[0]

    logging.info('loading predictions')
    uq_method_to_arrs_gen = get_method_to_arrs_gen(
        methods_whitelist=METHODS_WHITELIST,
        method_to_arr_names_dict=UQ_METHOD_TO_ARR_NAMES_DICT,
        io_helper=io_helper,
    )
    for method, arrs in uq_method_to_arrs_gen:
        logging.info(f'computing metrics for {method}:')
        if len(arrs) > 1:
            y_pred, y_quantiles, y_std = arrs
        else:
            y_pred, y_quantiles, y_std = arrs[0], None, None

        if SHORTEN_TO_TEST:
            y_pred, y_quantiles, y_std = _tail_shorten_arrs_none_ok(n_test_samples, y_pred, y_quantiles, y_std)

        metrics = {}
        logging.info(f'deterministic metrics...')
        metrics_det = compute_metrics_det(y_pred, y_true, metrics_whitelist=METRICS_WHITELIST_DET)
        if metrics_det:
            metrics.update(metrics_det)
        if len(arrs) > 1:
            logging.info(f'UQ metrics...')
            metrics_uq = compute_metrics_uq(y_pred, y_quantiles, y_std, y_true, settings.QUANTILES,
                                            metrics_whitelist=METRICS_WHITELIST_UQ)
            if metrics_uq:
                metrics.update(metrics_uq)
        logging.info(f'metrics: {metrics}')
        logging.info('saving metrics...')

        infix = 'test_' if SHORTEN_TO_TEST else ''
        filename = f'uq_metrics_{infix}{method}'
        if TIMESTAMPED_FILES:
            filename = misc_helpers.timestamped_filename(filename)
        io_helper.save_metrics(metrics, filename=filename)


def _tail_shorten_arrs_none_ok(lim, *arrs):
    for arr in arrs:
        yield arr[-lim:] if arr is not None else None


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


if __name__ == '__main__':
    main()
