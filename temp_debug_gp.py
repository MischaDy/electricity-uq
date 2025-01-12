import gpytorch
import numpy as np

import settings
from helpers.io_helper import IO_Helper
from make_partial_uq_plots import plot_uq_single_dataset
from src_uq_methods_native.gp_regression_gpytorch import ApproximateGP, prepare_data, predict_with_gpytorch

from helpers import misc_helpers


RUN_SIZE = 'full'
SHOW_PLOT = True
SAVE_PLOT = True
N_SAMPLES_TO_PLOT = 1600


def main():
    X_test, X_train, y_test, y_train = get_data()
    likelihood, model = load_models(X_train)
    interval = 90
    quantiles = settings.QUANTILES
    plot_train(X_train, interval, likelihood, model, quantiles, y_train, n_samples_to_plot=N_SAMPLES_TO_PLOT)
    plot_test(X_test, interval, likelihood, model, quantiles, y_test, n_samples_to_plot=N_SAMPLES_TO_PLOT)


def load_models(
        X_train,
        filename_lik='native_gpytorch_likelihood_n210432_it200.pth',
        filename_mod='native_gpytorch_n210432_it200.pth',
        n_inducing_points=500,
):
    io_helper = IO_Helper()
    likelihood = io_helper.load_torch_model_statedict(gpytorch.likelihoods.GaussianLikelihood,
                                                      filename=filename_lik)
    inducing_points = X_train[:n_inducing_points, :]
    model = io_helper.load_torch_model_statedict(
        ApproximateGP,
        filename=filename_mod,
        model_kwargs={'inducing_points': inducing_points},
    )
    return likelihood, model


def get_data():
    # noinspection PyProtectedMember
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers._quick_load_data(RUN_SIZE)
    X_train, y_train, X_val, y_val, _ = prepare_data(X_train, y_train, X_val, y_val, np.ones((1, 1)))
    return X_test, X_train, y_test, y_train


def plot_train(X_train, interval, likelihood, model, quantiles, y_train, n_samples_to_plot=1600):
    X_pred = X_train[:n_samples_to_plot]
    y_pred_train, y_quantiles_train, _ = predict_with_gpytorch(model, likelihood, X_pred, quantiles)
    np.save('temp_y_pred_train_gp.npy', y_pred_train)
    np.save('temp_y_quantiles_train_gp.npy', y_quantiles_train)
    plot_uq_single_dataset(y_train, y_pred_train, y_quantiles_train, uq_method='gp', is_training_data=True,
                           interval=interval, show_plot=SHOW_PLOT, save_plot=SAVE_PLOT,
                           n_samples_to_plot=n_samples_to_plot)


def plot_test(X_test, interval, likelihood, model, quantiles, y_test, n_samples_to_plot=1600):
    X_pred = X_test[:n_samples_to_plot]
    y_pred_test, y_quantiles_test, _ = predict_with_gpytorch(model, likelihood, X_pred, quantiles)
    np.save('temp_y_pred_test_gp.npy', y_pred_test)
    np.save('temp_y_quantiles_test_gp.npy', y_quantiles_test)
    plot_uq_single_dataset(y_test, y_pred_test, y_quantiles_test, uq_method='gp', is_training_data=True,
                           interval=interval, show_plot=SHOW_PLOT, save_plot=SAVE_PLOT,
                           n_samples_to_plot=n_samples_to_plot)


if __name__ == '__main__':
    main()
