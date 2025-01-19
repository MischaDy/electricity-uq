import logging

import seaborn as sns
from matplotlib import pyplot as plt

from helpers.io_helper import IO_Helper
from store_error_arrs import _get_filename

logging.basicConfig(level=logging.INFO, force=True)


RUN_SIZE = 'full'
SMALL_IO_HELPER = False
ARRAYS_FOLDER_BIG = 'arrays2'
MODELS_FOLDER_BIG = 'models'

BASE_MODEL_ERROR_SCORES = ['ae']
PROB_MODEL_ERROR_SCORES = ['crps', 'ae', 'ssr']

PLOT_HIST = True
PLOT_KDE = False
PLOT_HIST_WITH_KDE = False

PLOT_FOR_TEST = True
PLOT_FOR_TRAIN = False

SHOW_PLOTS = False
SAVE_PLOTS = True
SAVE_ARRAYS = True


METHODS_WHITELIST = {
    'base_model_hgbr',
    'base_model_linreg',
    'base_model_nn',
    'native_qhgbr',
    'native_qr',
    'native_gp',
    'native_mvnn',
    'posthoc_cp_hgbr',
    'posthoc_cp_linreg',
    'posthoc_cp_nn',
    'posthoc_la_nn',
}


METHOD_TO_ARR_NAMES_DICT = {
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
    assert (PLOT_FOR_TRAIN or PLOT_FOR_TEST)
    assert any([PLOT_KDE, PLOT_HIST, PLOT_HIST_WITH_KDE])

    if SMALL_IO_HELPER:
        io_helper_kwargs = {'arrays_folder': 'arrays_small', 'models_folder': 'models_small'}
    else:
        io_helper_kwargs = {'arrays_folder': ARRAYS_FOLDER_BIG, 'models_folder': MODELS_FOLDER_BIG}
    io_helper = IO_Helper(**io_helper_kwargs)

    dataset_to_plot_for = []
    if PLOT_FOR_TRAIN:
        dataset_to_plot_for.append('training')
    if PLOT_FOR_TEST:
        dataset_to_plot_for.append('test')
    for uq_method in METHODS_WHITELIST:
        error_scores = BASE_MODEL_ERROR_SCORES if uq_method.startswith('base_model') else PROB_MODEL_ERROR_SCORES
        for dataset in dataset_to_plot_for:
            for error_score in error_scores:
                logging.info(f"{uq_method=}, error score={error_score.upper()}, {dataset=}")
                filename = _get_filename(infix=error_score, uq_method=uq_method, dataset=dataset)
                error_arr_filename = f'{filename}.npy'
                try:
                    error_arr = io_helper.load_array(filename=error_arr_filename)
                except FileNotFoundError as e:
                    logging.info(f'error score array {e.filename} not found, skipping')
                    continue
                # noinspection PyUnboundLocalVariable
                plot_all(error_arr, io_helper, filename)


def plot_all(error_arr, io_helper, filename):
    if PLOT_HIST:
        logging.info('plotting histogram')
        plot_histogram(error_arr, io_helper, filename=f'{filename}_hist', bins=25)
    if PLOT_KDE:
        logging.info('plotting KDE')
        plot_kde(error_arr, io_helper, filename=f'{filename}_kde', bw_adjust=1)
    if PLOT_HIST_WITH_KDE:
        logging.info('plotting histogram with KDE')
        plot_hist_kde(error_arr, io_helper, filename=f'{filename}_histkde', bins=25)


def _plot_save_close(io_helper, filename):
    if SAVE_PLOTS:
        logging.info('saving plot')
        io_helper.save_plot(filename=filename)
    if SHOW_PLOTS:
        plt.show(block=True)
    plt.close()


def plot_histogram(error_arr, io_helper, filename, bins=25):
    sns.displot(error_arr, bins=bins)
    _plot_save_close(io_helper, filename)


def plot_kde(error_arr, io_helper, filename, bw_adjust=1):
    sns.displot(error_arr, kind="kde", bw_adjust=bw_adjust)
    _plot_save_close(io_helper, filename)


def plot_hist_kde(error_arr, io_helper, filename, bins=25):
    sns.displot(error_arr, kde=True, bins=bins)
    _plot_save_close(io_helper, filename)


def split_pred_arrs_train_test(arrs, n_samples_train):
    arrs_train = list(map(lambda arr: arr[:n_samples_train], arrs))
    arrs_test = list(map(lambda arr: arr[n_samples_train:], arrs))
    return arrs_train, arrs_test


if __name__ == '__main__':
    main()
