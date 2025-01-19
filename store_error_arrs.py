import logging
from functools import partial

from helpers.io_helper import IO_Helper
from helpers import misc_helpers
from helpers._metrics import crps, mae, ssr
from helpers.arr_helpers import get_method_to_arrs_gen


logging.basicConfig(level=logging.INFO, force=True)


RUN_SIZE = 'full'
COMPUTE_FOR_TRAIN = True
COMPUTE_FOR_TEST = True
SMALL_IO_HELPER = False
ARRAYS_FOLDER_BIG = 'arrays2'
MODELS_FOLDER_BIG = 'models'

METHODS_WHITELIST = {
    'base_model_hgbr',
    'base_model_linreg',
    'base_model_nn',
    'qhgbr',
    'qr',
    'gp',
    'mvnn',
    'cp_hgbr',
    'cp_linreg',
    'cp_nn',
    'la_nn',
}


METHOD_TO_ARR_NAMES_DICT = {
    'base_model_hgbr': ['base_model_hgbr_n210432_it30_its3.npy'],
    'base_model_linreg': ['base_model_linreg_n210432.npy'],
    'base_model_nn': ['base_model_nn_n210432_it400_nh2_hs50.npy'],
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
        'native_gpytorch_y_pred_n210432_it200_new.npy',
        'native_gpytorch_y_quantiles_n210432_it200_new.npy',
        'native_gpytorch_y_std_n210432_it200_new.npy',
    ],
    'mvnn': [
        'native_mvnn_y_pred_n210432_it100_nh2_hs50.npy',
        'native_mvnn_y_quantiles_n210432_it100_nh2_hs50.npy',
        'native_mvnn_y_std_n210432_it100_nh2_hs50.npy',
    ],
    'cp_hgbr': [
        'posthoc_conformal_prediction_base_model_hgbr_y_pred_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_hgbr_y_quantiles_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_hgbr_y_std_n210432_it5.npy',
    ],
    'cp_linreg': [
        'posthoc_conformal_prediction_base_model_linreg_y_pred_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_linreg_y_quantiles_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_linreg_y_std_n210432_it5.npy',
    ],
    'cp_nn': [
        'posthoc_conformal_prediction_base_model_nn_y_pred_n210432_it5_cp2.npy',
        'posthoc_conformal_prediction_base_model_nn_y_quantiles_n210432_it5_cp2.npy',
        'posthoc_conformal_prediction_base_model_nn_y_std_n210432_it5_cp2.npy',
    ],
    'la_nn': [
        'posthoc_laplace_approximation_base_model_nn_y_pred_n210432_it1000_la2.npy',
        'posthoc_laplace_approximation_base_model_nn_y_quantiles_n210432_it1000_la2.npy',
        'posthoc_laplace_approximation_base_model_nn_y_std_n210432_it1000_la2.npy',
    ],
}


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers._quick_load_data(RUN_SIZE)
    X_train, y_train = misc_helpers.add_val_to_train(X_train, X_val, y_train, y_val)
    y_train, y_test, y = misc_helpers.make_arrs_1d(y_train, y_test, y)

    n_samples_train = X_train.shape[0]

    if SMALL_IO_HELPER:
        io_helper_kwargs = {'arrays_folder': 'arrays_small', 'models_folder': 'models_small'}
    else:
        io_helper_kwargs = {'arrays_folder': ARRAYS_FOLDER_BIG, 'models_folder': MODELS_FOLDER_BIG}
    io_helper = IO_Helper(**io_helper_kwargs)
    uq_method_to_arrs_gen = get_method_to_arrs_gen(
        method_to_arr_names_dict=METHOD_TO_ARR_NAMES_DICT,
        methods_whitelist=METHODS_WHITELIST,
        io_helper=io_helper,
    )
    for method, uq_arrs in uq_method_to_arrs_gen:
        arrs_train, arrs_test = split_pred_arrs_train_test(uq_arrs, n_samples_train=n_samples_train)
        data_to_compute_for = []
        if COMPUTE_FOR_TRAIN:
            data_to_compute_for.append([y_train, arrs_train, True])
        if COMPUTE_FOR_TEST:
            data_to_compute_for.append([y_test, arrs_test, False])
        for y_true, arrs, are_train_arrs in data_to_compute_for:
            y_pred, y_quantiles, y_std = misc_helpers.make_arrs_1d(*arrs)
            y_true = misc_helpers.make_arr_1d(y_true)
            error_scores_dict = {
                'crps': partial(crps, y_true, y_quantiles, keep_dim=True),
                'ae': partial(mae, y_true, y_pred, keep_dim=True),
                'ssr': partial(ssr, y_true, y_pred, y_std, keep_dim=True),
            }
            dataset = 'training' if are_train_arrs else 'test'
            for error_score_name, error_func in error_scores_dict.items():
                logging.info(f"{method=}, error score={error_score_name.upper()}, {dataset=}")
                filename = _get_filename(infix=error_score_name, uq_method=method, dataset=dataset)
                logging.info(f'computing {error_score_name}')
                error_arr = error_func()
                logging.info('saving array')
                io_helper.save_array(error_arr, filename=filename)


def _get_filename(infix: str, uq_method: str, dataset: str, ext: str = None):
    """

    :param infix:
    :param uq_method:
    :param dataset: one of 'training', 'test'
    :param ext:
    :return:
    """
    filename_y_pred = METHOD_TO_ARR_NAMES_DICT[uq_method][0]
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
