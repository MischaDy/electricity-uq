import logging

import torch.nn
from matplotlib import pyplot as plt

from src_uq_methods_native.mean_var_nn import MeanVarNN, predict_with_mvnn

logging.basicConfig(level=logging.INFO, force=True)

from helpers import misc_helpers
from helpers.io_helper import IO_Helper


RUN_SIZE = 'big'
SMALL_IO_HELPER = True


arr_names = [
    'native_mvnn_y_pred_n35136_it150_nh2_hs50.npy',
    'native_mvnn_y_quantiles_n35136_it150_nh2_hs50.npy',
    'native_mvnn_y_std_n35136_it150_nh2_hs50.npy',
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
    io_helper = IO_Helper(arrays_folder='remote_error_arrays_mvnn')


model = io_helper.load_torch_model_statedict(
    MeanVarNN,
    filename='native_mvnn_n35136_it150_nh2_hs50.pth',
    model_kwargs={
        'dim_in': X_train.shape[1],
        'num_hidden_layers': 2,
        'hidden_layer_size': 50,
        'activation': torch.nn.LeakyReLU,
    }
)

X_pred = X_test
quantiles = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]

res = predict_with_mvnn(model, X_pred, quantiles)


# y_pred, y_quantiles, y_std = uq_arr_helpers.load_arrs(arr_names, io_helper=io_helper)
#
# n_samples_train = y_train.shape[0]
# n_samples_val = y_val.shape[0]
# y_pred_train, y_quantiles_train, y_std_train = map(lambda arr: arr[:n_samples_train], (y_pred, y_quantiles, y_std))
# y_pred_test, y_quantiles_test, y_std_test = map(lambda arr: arr[n_samples_train+n_samples_val:],
#                                                 (y_pred, y_quantiles, y_std))


y_train, y_test, y_val, y = misc_helpers.make_arrs_1d(y_train, y_test, y_val, y)


# p([y_test, y_pred_test], ['test', 'pred'])
