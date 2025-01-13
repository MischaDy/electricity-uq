import logging

import numpy as np
from matplotlib import pyplot as plt

import settings
from helpers.io_helper import IO_Helper
from helpers.misc_helpers import get_data
from helpers import uq_arr_helpers

logging.basicConfig(level=logging.INFO)


SAVE_PLOT = True
SHOW_PLOT = False
PLOT_EXT = 'svg'
METHOD_NAME_TO_BASE_FILENAME = {
    'qhgbr': 'native_qhgbr',
    'qr': 'native_quantile_regression_nn',
    'gp': 'native_gpytorch',
    'mvnn': 'native_mvnn',
    'cp_hgbr': 'posthoc_conformal_prediction_base_model_hgbr',
    'cp_linreg': 'posthoc_conformal_prediction_base_model_linreg',
    'cp_nn': 'posthoc_conformal_prediction_base_model_nn',
    'la_nn': 'posthoc_laplace_approximation_base_model_nn',
}

IO_HELPER = IO_Helper()


def main():
    from settings_update import update_run_size_setup

    logging.info('loading data')
    assert settings.RUN_SIZE in ['big', 'full']
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

    logging.info('loading predictions')
    uq_method_to_arrs_gen = uq_arr_helpers.get_uq_method_to_arrs_gen(io_helper=IO_HELPER)
    for uq_method, arrs in uq_method_to_arrs_gen:
        logging.info(f'plotting for method {uq_method}')
        y_pred, y_quantiles, y_std = arrs
        if np.any(y_pred < 0):
            logging.warning(f"arrays of {uq_method} don't seem to be scaled up properly, e.g.:"
                            f" {y_pred[0]=}; {y_quantiles[0, 0]=}; {y_std[0]=}. Rescaling them for plotting...")
            y_std = y_std * scaler_y.scale_
            y_pred, y_quantiles = map(scaler_y.inverse_transform, [y_pred.reshape(-1, 1), y_quantiles])
            y_pred = y_pred.squeeze()
        plot_uq(y_train, y_val, y_test, y_pred, y_quantiles, uq_method, interval=90, show_plot=SHOW_PLOT,
                save_plot=SAVE_PLOT, ext=PLOT_EXT)


def plot_uq_single_dataset(y_true, y_pred, y_quantiles, uq_method, interval: int | float, is_training_data,
                           n_samples_to_plot=1600, show_plot=True, save_plot=True, ext=None):
    n_quantiles = y_quantiles.shape[1]
    if n_quantiles == 99:
        ind_5p, ind_95p = 5-1, 95-1  # starts with 1
        ci_low, ci_high = y_quantiles[:, ind_5p], y_quantiles[:, ind_95p]
    else:
        raise ValueError("can't automatically determine which quantiles to plot")

    y_plot = y_true[:n_samples_to_plot]
    y_pred = y_pred[:n_samples_to_plot]
    ci_low = ci_low[:n_samples_to_plot]
    ci_high = ci_high[:n_samples_to_plot]
    plot_uq_worker(
        y_plot,
        y_pred,
        ci_low,
        ci_high,
        uq_method=uq_method,
        is_training_data=is_training_data,
        interval=interval,
        # n_stds=n_stds,
        show_plot=show_plot,
        save_plot=save_plot,
        ext=ext,
    )


def plot_uq(y_train, y_val, y_test, y_pred, y_quantiles, uq_method, interval: int | float, n_samples_to_plot=1600,
            show_plot=True, save_plot=True, ext=None):
    n_quantiles = y_quantiles.shape[1]
    if n_quantiles == 99:
        ind_5p, ind_95p = 5-1, 95-1  # starts with 1
        ci_low, ci_high = y_quantiles[:, ind_5p], y_quantiles[:, ind_95p]
    else:
        raise ValueError("can't automatically determine which quantiles to plot")
    # ci_low, ci_high = y_pred - n_stds * y_std, y_pred + n_stds * y_std

    # plot train
    y_train_plot = y_train[:n_samples_to_plot]
    y_pred_train = y_pred[:n_samples_to_plot]
    ci_low_train = ci_low[:n_samples_to_plot]
    ci_high_train = ci_high[:n_samples_to_plot]
    plot_uq_worker(
        y_train_plot,
        y_pred_train,
        ci_low_train,
        ci_high_train,
        uq_method=uq_method,
        is_training_data=True,
        interval=interval,
        # n_stds=n_stds,
        show_plot=show_plot,
        save_plot=save_plot,
        ext=ext,
    )

    # plot test
    start_test = y_train.shape[0] + y_val.shape[0]
    y_test_plot = y_test[:n_samples_to_plot]
    y_pred_test = y_pred[start_test: start_test + n_samples_to_plot]
    ci_low_test = ci_low[start_test: start_test + n_samples_to_plot]
    ci_high_test = ci_high[start_test: start_test + n_samples_to_plot]
    plot_uq_worker(
        y_test_plot,
        y_pred_test,
        ci_low_test,
        ci_high_test,
        uq_method=uq_method,
        is_training_data=False,
        interval=interval,
        # n_stds=n_stds,
        show_plot=show_plot,
        save_plot=save_plot,
        ext=ext,
    )


def plot_uq_worker(
        y_true_plot,
        y_pred_plot,
        ci_low_plot,
        ci_high_plot,
        uq_method,
        is_training_data: bool,
        interval: float | int = None,
        n_stds=None,
        plotting_quantiles=True,
        show_plot=True,
        save_plot=True,
        ext=None,
):
    """

    :param ext:
    :param y_true_plot:
    :param y_pred_plot:
    :param ci_low_plot:
    :param ci_high_plot:
    :param is_training_data:
    :param uq_method:
    :param interval: changes label. if plotting_quantiles, name the interval being plotted, e.g. "90%"
    :param n_stds: changes label.
    :param plotting_quantiles: changes label. True/False means quantiles/stds are being plotted, respectively.
    :param show_plot:
    :param save_plot:
    :return:
    """
    if not plotting_quantiles:
        logging.error('only plotting quantiles supported right now, not stds. trying to ignore and plot quantiles...')

    base_title = uq_method
    try:
        base_filename = METHOD_NAME_TO_BASE_FILENAME[uq_method]
    except KeyError:
        logging.warning(f"method {uq_method} not found in base filenames dict. using the method name as a base filename.")
        base_filename = uq_method
    dataset_label = 'training' if is_training_data else 'test'

    interval = int(interval) if float(interval).is_integer() else round(100 * interval, 2)
    uq_label = f'{interval}% CI' if plotting_quantiles else f'{n_stds} std'

    x_plot = np.arange(y_true_plot.shape[0])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    ax.plot(x_plot, y_true_plot, label=f'{dataset_label} data', color="black", linestyle='dashed')
    ax.plot(x_plot, y_pred_plot, label="point prediction", color="green")
    ax.fill_between(
        x_plot,
        ci_low_plot,
        ci_high_plot,
        color="green",
        alpha=0.2,
        label=uq_label,
    )
    ax.legend()
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title(f'{base_title} ({dataset_label})')
    if save_plot:
        filename = f'{base_filename}_{dataset_label}'
        if ext is not None:
            filename = f'{filename}.{ext}'
        IO_HELPER.save_plot(filename=filename)
    if show_plot:
        plt.show(block=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
