import logging
from functools import partial

import seaborn as sns
from matplotlib import pyplot as plt

from helpers.io_helper import IO_Helper
from helpers import misc_helpers
from helpers._metrics import crps, mae, ssr
from helpers.uq_arr_helpers import get_uq_method_to_arrs_gen


logging.basicConfig(level=logging.INFO, force=True)


RUN_SIZE = 'small'
SHOW_PLOTS = False
SAVE_PLOTS = True
SAVE_ARRAYS = True
RECOMPUTE_ERRORS = False
SMALL_IO_HELPER = True

UQ_METHODS_WHITELIST = {
    # 'qhgbr',
    # 'qr',
    # 'gp',
    # 'mvnn',
    'cp_hgbr',
    # 'cp_linreg',
    # 'cp_nn',
    # 'la_nn',
}

UQ_METHOD_TO_ARR_NAMES_DICT = {
    'qhgbr': [
        'native_qhgbr_y_pred_n210432_it0.npy',
        'native_qhgbr_y_quantiles_n210432_it0.npy',
        'native_qhgbr_y_std_n210432_it0.npy',
    ],
    'qr': [
        'native_quantile_regression_nn_y_pred_n210432_it300_nh2_hs50.npy',
        'native_quantile_regression_nn_y_quantiles_n210432_it300_nh2_hs50.npy',
        'native_quantile_regression_nn_y_std_n210432_it300_nh2_hs50.npy',
    ],
    'gp': [
        'native_gpytorch_y_pred_n210432_it200.npy',
        'native_gpytorch_y_quantiles_n210432_it200.npy',
        'native_gpytorch_y_std_n210432_it200.npy',
    ],
    'mvnn': [
        'native_mvnn_y_pred_n210432_it100_nh2_hs50.npy',
        'native_mvnn_y_quantiles_n210432_it100_nh2_hs50.npy',
        'native_mvnn_y_std_n210432_it100_nh2_hs50.npy',
    ],
    'cp_hgbr': [
        'posthoc_conformal_prediction_base_model_hgbr_y_pred_n640_it5.npy',
        'posthoc_conformal_prediction_base_model_hgbr_y_quantiles_n640_it5.npy',
        'posthoc_conformal_prediction_base_model_hgbr_y_std_n640_it5.npy',
    ],
    'cp_linreg': [
        'posthoc_conformal_prediction_base_model_linreg_y_pred_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_linreg_y_quantiles_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_linreg_y_std_n210432_it5.npy',
    ],
    'cp_nn': [
        'posthoc_conformal_prediction_base_model_nn_y_pred_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_nn_y_quantiles_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_nn_y_std_n210432_it5.npy',
    ],
    'la_nn': [
        'posthoc_laplace_approximation_base_model_nn_y_pred_n210432_it100.npy',
        'posthoc_laplace_approximation_base_model_nn_y_quantiles_n210432_it100.npy',
        'posthoc_laplace_approximation_base_model_nn_y_std_n210432_it100.npy',
    ],
}


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers._quick_load_data(RUN_SIZE)
    X_train, y_train = misc_helpers.add_val_to_train(X_train, X_val, y_train, y_val)
    y_train, y_test, y = misc_helpers.make_arrs_1d(y_train, y_test, y)

    n_samples_train = X_train.shape[0]

    io_helper_kwargs = {'arrays_folder': 'arrays_small', 'models_folder': 'models_small'} if SMALL_IO_HELPER else {}
    io_helper = IO_Helper(**io_helper_kwargs)
    uq_method_to_arrs_gen = get_uq_method_to_arrs_gen(
        uq_method_to_arr_names_dict=UQ_METHOD_TO_ARR_NAMES_DICT,
        uq_methods_whitelist=UQ_METHODS_WHITELIST,
        io_helper=io_helper,
    )
    recompute_errors = RECOMPUTE_ERRORS
    for uq_method, uq_arrs in uq_method_to_arrs_gen:
        arrs_train, arrs_test = split_pred_arrs_train_test(uq_arrs, n_samples_train=n_samples_train)
        for y_true, arrs, are_train_arrs in [(y_train, arrs_train, True), (y_test, arrs_test, False)]:
            y_pred, y_quantiles, y_std = arrs
            error_scores_dict = {
                'crps': partial(crps, y_true, y_quantiles, keep_dim=True),
                'ae': partial(mae, y_true, y_pred, keep_dim=True),
                'ssr': partial(ssr, y_true, y_pred, y_std, keep_dim=True),
            }
            dataset = 'training' if are_train_arrs else 'test'
            for error_score_name, error_func in error_scores_dict.items():
                logging.info(f"{uq_method=}, error score={error_score_name.upper()}, {dataset=}")
                filename = _get_filename(infix=error_score_name, uq_method=uq_method, dataset=dataset)
                if not recompute_errors:
                    error_arr_filename = f'{filename}.npy'
                    try:
                        error_arr = io_helper.load_array(filename=error_arr_filename)
                    except FileNotFoundError as e:
                        logging.info(f'error score array {e.filename} not found, computing from scratch')
                        recompute_errors = True

                if recompute_errors:
                    logging.info(f'computing {error_score_name}')
                    error_arr = error_func()
                    if SAVE_ARRAYS:
                        logging.info('saving array')
                        io_helper.save_array(error_arr, filename=filename)
                else:
                    logging.info('skipping error computation')

                logging.info('plotting histogram')
                # noinspection PyUnboundLocalVariable
                plot_histogram(error_arr)
                if SAVE_PLOTS:
                    logging.info('saving plot')
                    io_helper.save_plot(filename=filename)
                if SHOW_PLOTS:
                    plt.show(block=True)
                plt.close()


def plot_histogram(error_arr, bins=25):
    sns.displot(error_arr, bins=bins)


def _get_filename(infix: str, uq_method: str, dataset: str, ext: str = None):
    """

    :param infix:
    :param uq_method:
    :param dataset: one of 'training', 'test'
    :param ext:
    :return:
    """
    filename_y_pred = UQ_METHOD_TO_ARR_NAMES_DICT[uq_method][0]
    assert 'y_pred' in filename_y_pred
    base_filename = filename_y_pred.split('_y_pred_')[0]
    filename = f'{base_filename}_{infix}_{dataset}'
    if ext is not None:
        filename = f'{filename}.{ext}'
    return filename


def split_pred_arrs_train_test(arrs, n_samples_train):
    arrs_train = list(map(lambda arr: arr[:n_samples_train], arrs))
    arrs_test = list(map(lambda arr: arr[n_samples_train:], arrs))
    return arrs_train, arrs_test


if __name__ == '__main__':
    main()
