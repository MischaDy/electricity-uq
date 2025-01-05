import logging
import settings
logging.basicConfig(level=settings.LOGGING_LEVEL, force=True)
logging.info('importing')

import numpy as np
from matplotlib import pyplot as plt
from helpers import misc_helpers
from uq_comparison_pipeline import (
    update_run_size_setup, update_training_flags, update_progress_bar_settings,
    # UQ_Comparison_Pipeline, check_method_kwargs_dict
)
from helpers.io_helper import IO_Helper


def plot(X_train, X_val, X_test, y_train, y_val, y_test, y_pred, scaler_y):
    fig, ax = plt.subplots(figsize=(14, 6))
    start_train, end_train = 0, X_train.shape[0]
    start_val, end_val = end_train, end_train + X_val.shape[0]
    start_test, end_test = end_val, end_val + X_test.shape[0]
    x_plot_train = np.arange(start_train, end_train)
    x_plot_val = np.arange(start_val, end_val)
    x_plot_test = np.arange(start_test, end_test)
    x_plot_full = np.arange(start_train, end_test)
    if scaler_y is not None:
        y_train_, y_val_, y_test_ = misc_helpers.inverse_transform_ys(scaler_y, y_train, y_val, y_test)
    else:
        y_train_, y_val_, y_test_ = y_train, y_val, y_test
    ax.plot(x_plot_train, y_train_, label='train data', color="black", linestyle='dashed')
    ax.plot(x_plot_val, y_val_, label='val data', color="purple", linestyle='dashed')
    ax.plot(x_plot_test, y_test_, label='test data', color="blue", linestyle='dashed')
    ax.plot(x_plot_full, y_pred, label='y pred', color='green')
    ax.legend()
    plt.show(block=True)


FULL_DATA = False
SHOW_PLOT = True

logging.info('running preliminary checks/setup...')
# check_method_kwargs_dict(UQ_Comparison_Pipeline, settings.METHODS_KWARGS)
update_run_size_setup()
update_training_flags()
update_progress_bar_settings()
# uq_comparer = UQ_Comparison_Pipeline(
#     filename_parts=settings.FILENAME_PARTS,
#     data_path=settings.DATA_FILEPATH,
#     storage_path=settings.STORAGE_PATH,
#     methods_kwargs=settings.METHODS_KWARGS,
#     method_whitelist=settings.METHOD_WHITELIST,
#     n_points_per_group=settings.N_POINTS_PER_GROUP,
# )
logging.info('get data')

if FULL_DATA:
    input_cols = None
else:
    input_cols = [
        'ts_pred',
        'cat_is_sunday_or_holiday',
        'cat_is_saturday_and_not_holiday',
        'cat_is_workday',
        'cat_is_heating_period',
        'load_last_year',
        'load_last_week',
        'load_yesterday',
        'load_last_hour',
    ]

X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers.get_data(
    filepath='./data/data_2015_2018.pkl',
    train_years=settings.TRAIN_YEARS,
    val_years=settings.VAL_YEARS,
    test_years=settings.TEST_YEARS,
    input_cols=input_cols,
)

logging.info('load model')
io_helper = IO_Helper('comparison_storage')
filename = 'base_model_nn_n35136_it100_nh2_hs50_cpu.pth'
model = io_helper.load_model(filename=filename)
logging.info('predict')
y_pred = model.predict(X)

y_pred_orig_scale = scaler_y.inverse_transform(y_pred)
if SHOW_PLOT:
    logging.info('plot pred')
    x_plot = np.arange(y_pred_orig_scale.shape[0])
    plt.plot(x_plot, y_pred_orig_scale)
    plt.show(block=True)

sub_zeros = y_pred_orig_scale < 0
print('num sub zeros:', sub_zeros.sum())
sub_zeros_stacked = np.hstack([sub_zeros] * X.shape[1])
X_sub_zeros = X[sub_zeros_stacked].reshape(-1, 8)
print('info X_sub_zeros:')
print('mean:', X_sub_zeros.mean(axis=0))
print('min:', X_sub_zeros.min(axis=0))
print('max:', X_sub_zeros.max(axis=0))
print('std:', X_sub_zeros.std(axis=0))

logging.info('pred subzero')
try:
    X_pred_sub_zero = X_sub_zeros[285:286]
except IndexError:
    X_pred_sub_zero = X_sub_zeros[:1]
y_pred_sub_zero = model.predict(X_pred_sub_zero)
print(f'pred for X = {X_pred_sub_zero}:\ny = {y_pred_sub_zero}')

logging.info('plot full')
plot(X_train, X_val, X_test, y_train, y_val, y_test, y_pred, scaler_y)
