import logging

from matplotlib import pyplot as plt
logging.basicConfig(level=logging.INFO, force=True)

from helpers import uq_arr_helpers, misc_helpers
from helpers.io_helper import IO_Helper


RUN_SIZE = 'full'
SMALL_IO_HELPER = False


arr_names = [
    # 'posthoc_conformal_prediction_base_model_hgbr_y_pred_n640_it5.npy',
    # 'posthoc_conformal_prediction_base_model_hgbr_y_quantiles_n640_it5.npy',
    # 'posthoc_conformal_prediction_base_model_hgbr_y_std_n640_it5.npy',

    'posthoc_conformal_prediction_base_model_hgbr_y_pred_n210432_it5.npy',
    'posthoc_conformal_prediction_base_model_hgbr_y_quantiles_n210432_it5.npy',
    'posthoc_conformal_prediction_base_model_hgbr_y_std_n210432_it5.npy',
]


def p(arrs, labels=None):
    for i, arr in enumerate(arrs):
        plt.plot(arr, label=f'arr{i}' if labels is None else labels[i])
    plt.legend()
    plt.show(block=True)


def s():
    plt.show(block=True)


X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers._quick_load_data(RUN_SIZE)

if SMALL_IO_HELPER:
    io_helper = IO_Helper(arrays_folder='arrays_small', models_folder='models_small')
else:
    io_helper = IO_Helper()

y_pred, y_quantiles, y_std = uq_arr_helpers.load_arrs(arr_names, io_helper=io_helper)

n_samples_train = y_train.shape[0]
n_samples_val = y_val.shape[0]
y_pred_train, y_quantiles_train, y_std_train = map(lambda arr: arr[:n_samples_train], (y_pred, y_quantiles, y_std))
y_pred_test, y_quantiles_test, y_std_test = map(lambda arr: arr[n_samples_train+n_samples_val:],
                                                (y_pred, y_quantiles, y_std))

# y_pred, y_pred_train, y_pred_test, y_std, y_std_train, y_std_test = misc_helpers.make_arrs_2d(
#     y_pred, y_pred_train, y_pred_test, y_std, y_std_train, y_std_test
# )
y_train, y_test, y_val, y = misc_helpers.make_arrs_1d(y_train, y_test, y_val, y)


# p([y_test, y_pred_test], ['test', 'pred'])
