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
SAVE_PREDS = True
N_SAMPLES_TO_PLOT = 1600


def main():
    X_test, X_train, y_test, y_train, scaler_y = get_data()
    likelihood, model = load_models(X_train)

    for X_pred, y_true, is_training_data in [(X_train, y_train, True), (X_test, y_test, False)]:
        X_pred = X_pred[:N_SAMPLES_TO_PLOT]
        y_pred, y_quantiles = predict(X_pred, model, likelihood, settings.QUANTILES, is_training_data=is_training_data,
                                      save_preds=SAVE_PREDS)
        y_pred, y_quantiles = misc_helpers.inverse_transform_ys(scaler_y, y_pred, y_quantiles)
        plot(y_true, y_pred, y_quantiles, interval=90, is_training_data=is_training_data,
             n_samples_to_plot=N_SAMPLES_TO_PLOT)


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
    return X_test, X_train, y_test, y_train, scaler_y


def predict(X_pred, model, likelihood, quantiles, is_training_data, save_preds=True):
    y_pred, y_quantiles, _ = predict_with_gpytorch(model, likelihood, X_pred, quantiles)
    if save_preds:
        infix = 'train' if is_training_data else 'test'
        np.save(f'temp_y_pred_{infix}_gp.npy', y_pred)
        np.save(f'temp_y_quantiles_{infix}_gp.npy', y_quantiles)
    return y_pred, y_quantiles


def plot(y_true, y_pred, y_quantiles, interval, is_training_data, n_samples_to_plot=1600):
    plot_uq_single_dataset(y_true, y_pred, y_quantiles, uq_method='gp', is_training_data=is_training_data,
                           interval=interval, show_plot=SHOW_PLOT, save_plot=SAVE_PLOT,
                           n_samples_to_plot=n_samples_to_plot)


if __name__ == '__main__':
    main()
