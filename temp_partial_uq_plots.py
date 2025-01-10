from typing import Literal

import numpy as np
from matplotlib import pyplot as plt

import settings
from helpers.io_helper import IO_Helper
from helpers.misc_helpers import get_data

SAVE_PLOT = True
SHOW_PLOT = True

IO_HELPER = IO_Helper('comparison_storage')


def main():
    from uq_comparison_pipeline import update_run_size_setup

    update_run_size_setup()
    data = get_data(
        filepath=settings.DATA_FILEPATH,
        train_years=settings.TRAIN_YEARS,
        val_years=settings.VAL_YEARS,
        test_years=settings.TEST_YEARS,
        do_standardize_data=True,
    )
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = data
    y_train, y_val, y_test, y = map(scaler_y.inverse_transform, [y_train, y_val, y_test, y])

    all_arrs = get_arrays()
    for method_name_arrs, arrs in all_arrs.items():
        method = method_name_arrs.replace('_arrs', '')
        y_pred, y_quantiles, y_std = arrs
        plot_uq(y_train, y_val, y_test, y_pred, y_quantiles, method, show_plot=SHOW_PLOT, save_plot=SAVE_PLOT)


def get_arrays():
    qr_arrs = [
        np.load('remote_arrays_4/native_quantile_regression_nn_y_pred_n210432_it300_nh2_hs50.npy'),
        np.load('remote_arrays_4/native_quantile_regression_nn_y_quantiles_n210432_it300_nh2_hs50.npy'),
        np.load('remote_arrays_4/native_quantile_regression_nn_y_std_n210432_it300_nh2_hs50.npy'),
    ]
    gp_arrs = to_arrs([
        'native_gpytorch_y_pred_n210432_it200.npy',
        'native_gpytorch_y_quantiles_n210432_it200.npy',
        'native_gpytorch_y_std_n210432_it200.npy',
    ])
    mvnn_arrs = to_arrs([
        'native_mvnn_y_pred_n210432_it100_nh2_hs50.npy',
        'native_mvnn_y_quantiles_n210432_it100_nh2_hs50.npy',
        'native_mvnn_y_std_n210432_it100_nh2_hs50.npy',
    ])
    cp_hgbr_arrs = to_arrs([
        'posthoc_conformal_prediction_base_model_hgbr_y_pred_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_hgbr_y_quantiles_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_hgbr_y_std_n210432_it5.npy'
    ])
    cp_linreg_arrs = to_arrs([
        'posthoc_conformal_prediction_base_model_linreg_y_pred_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_linreg_y_quantiles_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_linreg_y_std_n210432_it5.npy'
    ])
    cp_nn_arrs = to_arrs([
        'posthoc_conformal_prediction_base_model_nn_y_pred_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_nn_y_quantiles_n210432_it5.npy',
        'posthoc_conformal_prediction_base_model_nn_y_std_n210432_it5.npy'
    ])
    la_nn_arrs = to_arrs([
        'posthoc_laplace_approximation_base_model_nn_y_pred_n210432_it100.npy',
        'posthoc_laplace_approximation_base_model_nn_y_quantiles_n210432_it100.npy',
        'posthoc_laplace_approximation_base_model_nn_y_std_n210432_it100.npy',
    ])

    all_arrs = {
        # 'gp_arrs': gp_arrs,
        # 'mvnn_arrs': mvnn_arrs,
        'qr_arrs': qr_arrs,
        # 'cp_hgbr_arrs': cp_hgbr_arrs,
        # 'cp_linreg_arrs': cp_linreg_arrs,
        # 'cp_nn_arrs': cp_nn_arrs,
        # 'la_nn_arrs': la_nn_arrs,
    }
    return all_arrs


def to_arrs(filenames):
    return [IO_HELPER.load_array(filename=filename) for filename in filenames]


def plot_uq(y_train, y_val, y_test, y_pred, y_quantiles, method, n_samples_to_plot=1600, show_plot=True,
            save_plot=True):
    ci_low, ci_high = (
        y_quantiles[:, 0],
        y_quantiles[:, -1],
    )
    # ci_low, ci_high = y_pred - n_stds * y_std, y_pred + n_stds * y_std
    n_quantiles = y_quantiles.shape[1]

    # plot train
    y_train_plot = y_train[:n_samples_to_plot]
    y_pred_train = y_pred[:n_samples_to_plot]
    ci_low_train = ci_low[:n_samples_to_plot]
    ci_high_train = ci_high[:n_samples_to_plot]
    plot_uq_worker(y_train_plot, y_pred_train, ci_low_train, ci_high_train, 'train', method, n_quantiles)

    # plot test
    start_test = y_train.shape[0] + y_val.shape[0]
    y_test_plot = y_test[:n_samples_to_plot]
    y_pred_test = y_pred[start_test: start_test + n_samples_to_plot]
    ci_low_test = ci_low[start_test: start_test + n_samples_to_plot]
    ci_high_test = ci_high[start_test: start_test + n_samples_to_plot]
    plot_uq_worker(y_test_plot, y_pred_test, ci_low_test, ci_high_test, 'test', method, n_quantiles,
                   show_plot=show_plot, save_plot=save_plot)


def plot_uq_worker(y_true_plot, y_pred_plot, ci_low_plot, ci_high_plot, train_or_test: Literal['train', 'test'],
                   method, n_quantiles, show_plot=True, save_plot=True):
    base_title = method
    base_filename = method
    label = f'outermost 2/{n_quantiles} quantiles'
    x_plot = np.arange(y_true_plot.shape[0])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    ax.plot(x_plot, y_true_plot, label=f'{train_or_test} data', color="black", linestyle='dashed')
    ax.plot(x_plot, y_pred_plot, label="point prediction", color="green")
    ax.fill_between(
        x_plot,
        ci_low_plot,
        ci_high_plot,
        color="green",
        alpha=0.2,
        label=label,
    )
    ax.legend()
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title(f'{base_title} ({train_or_test})')
    if save_plot:
        IO_HELPER.save_plot(filename=f'{base_filename}_{train_or_test}')
    if show_plot:
        plt.show(block=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
