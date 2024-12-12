#!/usr/bin/env python
# coding: utf-8

"""
source: https://mapie.readthedocs.io/en/latest/examples_regression/4-tutorials/plot_ts-tutorial.html




Before estimating prediction intervals with MAPIE, we optimize the base model,
here a Random Forest model. The hyperparameters are
optimized with a :class:`~sklearn.model_selection.RandomizedSearchCV` using a
sequential :class:`~sklearn.model_selection.TimeSeriesSplit` cross validation,
in which the training set is prior to the validation set.

Once the base model is optimized, we can use
:class:`~MapieTimeSeriesRegressor` to estimate
the prediction intervals associated with one-step ahead forecasts through
the EnbPI method.

As its parent class :class:`~MapieRegressor`,
:class:`~MapieTimeSeriesRegressor` has two main arguments : "cv", and "method".
In order to implement EnbPI, "method" must be set to "enbpi" (the default
value) while "cv" must be set to the :class:`~mapie.subsample.BlockBootstrap`
class that block bootstraps the training set.
This sampling method is used instead of the traditional bootstrap
strategy as it is more suited for time series data.

The EnbPI method allows you update the residuals during the prediction,
each time new observations are available so that the deterioration of
predictions, or the increase of noise level, can be dynamically taken into
account. It can be done with :class:`~MapieTimeSeriesRegressor` through
the ``partial_fit`` class method called at every step.


The ACI strategy allows you to adapt the conformal inference
(i.e. the quantile). If the real values are not in the coverage,
the size of the intervals will grow.
Conversely, if the real values are in the coverage,
the size of the intervals will decrease.
You can use a gamma coefficient to adjust the strength of the correction.

"""

import warnings
from typing import Iterable

import numpy as np
import numpy.typing as npt

from matplotlib import pylab as plt
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from mapie.metrics import (
    coverage_width_based,
    regression_coverage_score,
    regression_mean_width_score,
)
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap

from helpers import get_data
from io_helper import IO_Helper

warnings.simplefilter("ignore")

# todo: remove
N_POINTS_TEMP = 100  # per group

IO_HELPER = IO_Helper("mapie_storage")


def main(io_helper):
    print("loading data")
    X_train, X_test, y_train, y_test = get_data(N_POINTS_TEMP)
    print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    print("plotting data")
    plot_data(X_train, X_test, y_train, y_test, io_helper=io_helper)

    print("training base model")
    model_params_choices = {
        "max_depth": randint(2, 30),
        "n_estimators": randint(10, 100),
    }
    model = train_base_model(
        RandomForestRegressor,
        model_params_choices=model_params_choices,
        X_train=X_train,
        y_train=y_train,
        skip_training=True,
        io_helper=io_helper,
    )

    # print('estimating prediction intervals')
    # estimate_prediction_intervals_all(model, X_train, y_train, X_test, y_test)
    print("estimating prediction intervals enbpi without partial fit")
    estimate_prediction_intervals_enbpi_nopfit_all_quantiles(
        model, X_train, y_train, X_test, y_test, io_helper=io_helper,
    )


def train_base_model(
    model_class,
    model_params_choices=None,
    model_init_params=None,
    X_train=None,
    y_train=None,
    skip_training=True,
    cv_n_iter=100,
    n_jobs=-1,
    io_helper: IO_Helper = None,
):
    """Optimize the base estimator

    Before estimating the prediction intervals with MAPIE, let's optimize the
    base model, here a :class:`~RandomForestRegressor` through a
    :class:`~RandomizedSearchCV` with a temporal cross-validation strategy.
    For the sake of computational time, the best parameters are already tuned.
    """
    random_seed = 42
    if model_init_params is None:
        model_init_params = {}
    elif "random_seed" not in model_init_params:
        model_init_params["random_seed"] = random_seed

    filename_base_model = f"base_{model_class.__name__}_{N_POINTS_TEMP}.model"

    if skip_training:
        # Model previously optimized with a cross-validation:
        # RandomForestRegressor(max_depth=13, n_estimators=89, random_state=42)
        try:
            model = io_helper.load_model(filename_base_model)
            return model
        except FileNotFoundError:
            print(f"trained base model '{filename_base_model}' not found")

    assert all(item is not None for item in [X_train, y_train, model_params_choices])
    print("training")

    # CV parameter search
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = model_class(random_state=random_seed, **model_init_params)
    cv_obj = RandomizedSearchCV(
        model,
        param_distributions=model_params_choices,
        n_iter=cv_n_iter,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=random_seed,
        verbose=1,
        n_jobs=n_jobs,
    )
    # todo: ravel?
    cv_obj.fit(X_train, y_train.values.ravel())
    model = cv_obj.best_estimator_
    print("done")
    io_helper.save_model(model, filename_base_model)
    return model


### PREDICTION INTERVALS


# noinspection PyPep8Naming
def estimate_prediction_intervals_all(model, X_train, y_train, X_test, y_test, io_helper: IO_Helper = None):
    """
    Estimate prediction intervals on test set

    We now use :class:`~MapieTimeSeriesRegressor` to build prediction intervals
    associated with one-step ahead forecasts. As explained in the introduction,
    we use the EnbPI method and the ACI method.

    Estimating prediction intervals can be possible in three ways:

    - with a regular ``.fit`` and ``.predict`` process, limiting the use of
      trainining set residuals to build prediction intervals

    - using ``.partial_fit`` in addition to ``.fit`` and ``.predict`` allowing
      MAPIE to use new residuals from the test points as new data are becoming
      available.

    - using ``.partial_fit`` and ``.adapt_conformal_inference`` in addition to
      ``.fit`` and ``.predict`` allowing MAPIE to use new residuals from the
      test points as new data are becoming available.

    The latter method is particularly useful to adjust prediction intervals to
    sudden change points on test sets that have not been seen by the model
    during training.

    We use the :class:`~BlockBootstrap` sampling
    method instead of the traditional bootstrap strategy for training the model
    since the former is more suited for time series data.
    Here, we choose to perform 10 resamplings with 10 blocks.
    """

    alpha = 0.05
    gap = 1
    skip_base_training = True
    save_trained = True
    skip_adaptation = True

    cv_mapie_ts = BlockBootstrap(n_resamplings=10, n_blocks=10, overlapping=False, random_state=42)

    print("\n===== estimating PIs no_pfit_enbpi")
    y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit = estimate_pred_interals_no_pfit_enbpi(
        model,
        cv_mapie_ts,
        alpha,
        X_test,
        X_train,
        y_train,
        skip_training=skip_base_training,
        save_trained=save_trained,
        io_helper=io_helper,
    )
    y_pis_enbpi_no_pfit = y_pis_enbpi_no_pfit.squeeze()

    print("\n===== estimating PIs no_pfit_aci")
    y_pred_aci_no_pfit, y_pis_aci_no_pfit = estimate_pred_interals_no_pfit_aci(
        model,
        cv_mapie_ts,
        y_pred_enbpi_no_pfit,
        y_pis_enbpi_no_pfit,
        alpha,
        gap,
        X_test,
        y_test,
        X_train,
        y_train,
        skip_base_training=skip_base_training,
        skip_adaptation=skip_adaptation,
        io_helper=io_helper,
    )
    y_pis_aci_no_pfit = y_pis_aci_no_pfit.squeeze()

    print("\n===== estimating PIs pfit_enbpi")
    y_pred_enbpi_pfit, y_pis_enbpi_pfit = estimate_pred_interals_pfit_enbpi(
        model,
        cv_mapie_ts,
        y_pred_enbpi_no_pfit,
        y_pis_enbpi_no_pfit,
        alpha,
        gap,
        X_train,
        y_train,
        X_test,
        y_test,
        skip_base_training=skip_base_training,
        skip_adaptation=skip_adaptation,
        io_helper=io_helper
    )
    y_pis_enbpi_pfit = y_pis_enbpi_pfit.squeeze()

    print("\n===== estimating PIs pfit_aci")
    y_pred_aci_pfit, y_pis_aci_pfit = estimate_pred_interals_pfit_aci(
        model,
        cv_mapie_ts,
        y_pred_aci_no_pfit,
        y_pis_aci_no_pfit,
        alpha,
        gap,
        X_train,
        y_train,
        X_test,
        y_test,
        skip_base_training=skip_base_training,
        skip_adaptation=skip_adaptation,
        io_helper=io_helper
    )
    y_pis_aci_pfit = y_pis_aci_pfit.squeeze()

    print("\n===== comparing coverages")
    compare_coverages(
        y_test, y_pis_aci_no_pfit, y_pis_aci_pfit, y_pis_enbpi_no_pfit, y_pis_enbpi_pfit, io_helper=io_helper
    )

    print("\n===== plotting prediction intervals")
    plot_prediction_intervals(
        y_train,
        y_test,
        y_pred_enbpi_no_pfit,
        y_pred_enbpi_pfit,
        y_pis_enbpi_no_pfit,
        y_pis_enbpi_pfit,
        y_pred_aci_no_pfit,
        y_pred_aci_pfit,
        y_pis_aci_no_pfit,
        y_pis_aci_pfit,
        io_helper=io_helper,
    )


# noinspection PyPep8Naming
def estimate_prediction_intervals_enbpi_nopfit_all_quantiles(
    model, X_train, y_train, X_test, y_test, io_helper: IO_Helper = None,
):
    alpha = np.linspace(0.01, 0.99, num=99, endpoint=True)
    skip_base_training = True

    cv_mapie_ts = BlockBootstrap(n_resamplings=10, n_blocks=10, overlapping=False, random_state=42)

    print("\n===== estimating PIs no_pfit_enbpi")
    y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit = estimate_pred_interals_no_pfit_enbpi(
        model,
        cv_mapie_ts,
        alpha,
        X_test,
        X_train,
        y_train,
        skip_training=skip_base_training,
        io_helper=io_helper,
    )

    print("\n===== plotting prediction intervals")
    plot_prediction_intervals_all_quantiles(
        y_train, y_test, y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit, io_helper=io_helper
    )


# noinspection PyPep8Naming
def estimate_pred_interals_no_pfit_enbpi(
    model,
    cv_mapie_ts,
    alpha: float | Iterable[float],
    X_test,
    X_train=None,
    y_train=None,
    skip_training=True,
    save_trained=True,
    io_helper: IO_Helper = None,
    agg_function: str = 'mean',
    verbose=1,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Estimate prediction intervals without partial fit using EnbPI.

    :param save_trained:
    :param verbose:
    :param agg_function:
    :param io_helper:
    :param model:
    :param cv_mapie_ts:
    :param alpha:
    :param X_test:
    :param X_train:
    :param y_train:
    :param skip_training:
    :return: tuple (y_pred, y_prediction_intervals) of shapes (n_samples,) and (n_samples, 2, n_alpha).
    [:, 0, alpha]: Lower bound of the prediction interval corresp. to confidence level alpha.
    [:, 1, alpha]: Upper bound of the prediction interval corresp. to confidence level alpha.
    """
    # todo: return type taken from mapie
    try:
        alpha = list(alpha)
    except TypeError:
        pass

    mapie_enbpi = MapieTimeSeriesRegressor(
        model, method="enbpi", cv=cv_mapie_ts, agg_function=agg_function, n_jobs=-1, verbose=verbose,
    )

    n_training_points = X_train.shape[0]
    n_test_points = X_test.shape[0]
    filename_enbpi_no_pfit = f"mapie_enbpi_no_pfit_{n_training_points}_{n_test_points}.model"

    # import ipdb
    # ipdb.set_trace()

    if skip_training:
        try:
            mapie_enbpi = io_helper.load_model(filename_enbpi_no_pfit)
            print("loaded model successfully")
        except FileNotFoundError:
            print(f"skipping training not possible")
            skip_training = False

    if not skip_training:
        print("training enbpi_no_pfit...")
        mapie_enbpi = mapie_enbpi.fit(X_train, y_train)

    if save_trained:
        io_helper.save_model(mapie_enbpi, filename_enbpi_no_pfit)

    print("predicting enbpi_no_pfit...")
    y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit = mapie_enbpi.predict(
        X_test, alpha=alpha, ensemble=False, optimize_beta=False, allow_infinite_bounds=True,
    )
    return y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit


# noinspection PyPep8Naming
def estimate_pred_interals_no_pfit_aci(
    model,
    cv_mapie_ts,
    y_pred_enbpi_no_pfit,
    y_pis_enbpi_no_pfit,
    alpha,
    gap,
    X_test,
    y_test,
    X_train=None,
    y_train=None,
    skip_base_training=True,
    skip_adaptation=True,
    io_helper: IO_Helper = None,
):
    """estimate prediction intervals without partial fit, ACI."""
    return _estimate_prediction_intervals_worker(
        model,
        cv_mapie_ts,
        y_pred_enbpi_no_pfit.shape,
        y_pis_enbpi_no_pfit.shape,
        alpha,
        gap,
        X_test,
        y_test,
        method="aci",
        with_partial_fit=False,
        X_train=X_train,
        y_train=y_train,
        skip_base_training=skip_base_training,
        skip_adaptation=skip_adaptation,
        io_helper=io_helper,
    )


# noinspection PyPep8Naming
def estimate_pred_interals_pfit_enbpi(
    model,
    cv_mapie_ts,
    y_pred_enbpi_no_pfit,
    y_pis_enbpi_no_pfit,
    alpha,
    gap,
    X_train,
    y_train,
    X_test,
    y_test,
    skip_base_training=True,
    skip_adaptation=True,
    io_helper: IO_Helper = None,
):
    """
    estimate prediction intervals with partial fit.
    The update of the residuals and the one-step ahead predictions are performed sequentially in a loop.
    """
    return _estimate_prediction_intervals_worker(
        model,
        cv_mapie_ts,
        y_pred_enbpi_no_pfit.shape,
        y_pis_enbpi_no_pfit.shape,
        alpha,
        gap,
        X_test,
        y_test,
        method="enbpi",
        with_partial_fit=True,
        X_train=X_train,
        y_train=y_train,
        skip_base_training=skip_base_training,
        skip_adaptation=skip_adaptation,
        io_helper=io_helper,
    )


# noinspection PyPep8Naming
def estimate_pred_interals_pfit_aci(
    model,
    cv_mapie_ts,
    y_pred_aci_no_pfit,
    y_pis_aci_no_pfit,
    alpha,
    gap,
    X_train,
    y_train,
    X_test,
    y_test,
    skip_base_training=True,
    skip_adaptation=True,
    io_helper: IO_Helper = None,
):
    """
    estimate prediction intervals with adapt_conformal_inference.
    As discussed previously, the update of the current alpha and the one-step
    ahead predictions are performed sequentially in a loop.
    """
    return _estimate_prediction_intervals_worker(
        model,
        cv_mapie_ts,
        y_pred_aci_no_pfit.shape,
        y_pis_aci_no_pfit.shape,
        alpha,
        gap,
        X_test,
        y_test,
        method="aci",
        with_partial_fit=True,
        X_train=X_train,
        y_train=y_train,
        skip_base_training=skip_base_training,
        skip_adaptation=skip_adaptation,
        io_helper=io_helper,
    )


# noinspection PyPep8Naming
def _estimate_prediction_intervals_worker(
    model,
    cv_mapie_ts,
    y_pred_shape,
    y_pis_shape,
    alpha,
    gap,
    X_test,
    y_test,
    method,
    with_partial_fit,
    X_train=None,
    y_train=None,
    skip_base_training=True,
    skip_adaptation=True,
    io_helper: IO_Helper = None,
):
    """overarching function for estimating prediction intervals"""
    pfit_str = "pfit" if with_partial_fit else "no_pfit"
    filename_model_base = f"mapie_{method}_{N_POINTS_TEMP}.model"
    filename_model_adapted = f"model_{method}_{pfit_str}_{N_POINTS_TEMP}.model"
    filename_arr_y_pred = f"y_pred_{method}_{pfit_str}_{N_POINTS_TEMP}.npy"
    filename_arr_y_pis = f"y_pis_{method}_{pfit_str}_{N_POINTS_TEMP}.npy"

    if skip_adaptation:
        try:
            y_pred = io_helper.load_array(filename_arr_y_pred)
            y_pis = io_helper.load_array(filename_arr_y_pis)
            return y_pred, y_pis
        except FileNotFoundError:
            print("skipping adaptation not possible")

    mapie_ts_regressor = MapieTimeSeriesRegressor(
        model, method=method, cv=cv_mapie_ts, agg_function="mean", n_jobs=-1
    )

    if skip_base_training:
        try:
            mapie_ts_regressor = io_helper.load_model(filename_model_base)
            print("loaded model successfully")
        except FileNotFoundError:
            print(f"skipping training not possible")
            skip_base_training = False

    if not skip_base_training:
        print("training mapie_ts_regressor...")
        mapie_ts_regressor = mapie_ts_regressor.fit(X_train, y_train)
        io_helper.save_model(mapie_ts_regressor, filename_model_base)

    y_pred = np.zeros(y_pred_shape)
    y_pis = np.zeros(y_pis_shape)

    print("predicting...")
    y_pred[:gap], y_pis[:gap, :, :] = mapie_ts_regressor.predict(
        X_test.iloc[:gap, :],
        alpha=alpha,
        ensemble=True,
        optimize_beta=True,
        allow_infinite_bounds=True,
    )

    print("looping...")
    eps = -1
    for step in range(gap, len(X_test), gap):
        if step % 10 == 0:
            print("step", step)
        if with_partial_fit:
            mapie_ts_regressor.partial_fit(
                X_test.iloc[(step - gap): step, :],
                y_test.iloc[(step - gap): step],
            )
        if method == "aci":
            mapie_ts_regressor.adapt_conformal_inference(
                X_test.iloc[(step - gap): step, :].to_numpy(),
                y_test.iloc[(step - gap): step].to_numpy(),
                gamma=0.05,
            )
        (
            y_pred[step: step + gap],
            y_pis[step: step + gap, :, :],
        ) = mapie_ts_regressor.predict(
            X_test.iloc[step: (step + gap), :],
            alpha=alpha,
            ensemble=True,
            optimize_beta=True,
            allow_infinite_bounds=True,
        )
        arr = y_pis[step: step + gap, :, :]
        arr[np.isinf(arr)] = eps

    io_helper.save_array(filename_arr_y_pred, y_pred)
    io_helper.save_array(filename_arr_y_pis, y_pis)
    io_helper.save_model(mapie_ts_regressor, filename_model_adapted)

    return y_pred, y_pis


### COMPUTE SCORES


def compute_scores(y_pis, y_test, eta):
    print("computing scores")
    coverage = regression_coverage_score(y_test, y_pis[:, 0, 0], y_pis[:, 1, 0])
    width = regression_mean_width_score(y_pis[:, 0, 0], y_pis[:, 1, 0])
    cwc = coverage_width_based(
        y_test, y_pis[:, 0, 0], y_pis[:, 1, 0], eta=eta, alpha=0.05
    )
    return coverage, width, cwc


### PLOTTING


# noinspection PyPep8Naming
def plot_data(X_train, X_test, y_train, y_test, filename="data", do_save_figure=True, io_helper: IO_Helper = None,):
    """visualize training and test sets"""
    num_train_steps = X_train.shape[0]
    num_test_steps = X_test.shape[0]

    x_plot_train = np.arange(num_train_steps)
    x_plot_test = x_plot_train + num_test_steps

    plt.figure(figsize=(16, 5))
    plt.plot(x_plot_train, y_train)
    plt.plot(x_plot_test, y_test)
    plt.ylabel("energy data (details TODO)")
    plt.legend(["Training data", "Test data"])
    if do_save_figure:
        io_helper.save_plot(f"{filename}_{N_POINTS_TEMP}.png")
    plt.show()


def plot_prediction_intervals(
    y_train,
    y_test,
    y_pred_enbpi_no_pfit,
    y_pred_enbpi_pfit,
    y_pis_enbpi_no_pfit,
    y_pis_enbpi_pfit,
    y_pred_aci_no_pfit,
    y_pred_aci_pfit,
    y_pis_aci_no_pfit,
    y_pis_aci_pfit,
    filename="prediction_intervals",
    io_helper: IO_Helper = None,
):
    """
    Plot estimated prediction intervals on one-step ahead forecast

    compare the prediction intervals estimated by MAPIE with and
    without update of the residuals.
    """
    coverage_enbpi_no_pfit, width_enbpi_no_pfit, cwc_enbpi_no_pfit = compute_scores(
        y_pis_enbpi_no_pfit, y_test, eta=10
    )
    coverage_aci_no_pfit, width_aci_no_pfit, cwc_aci_no_pfit = compute_scores(
        y_pis_aci_no_pfit, y_test, eta=10
    )
    coverage_enbpi_pfit, width_enbpi_pfit, cwc_enbpi_pfit = compute_scores(
        y_pis_enbpi_pfit, y_test, eta=10
    )
    coverage_aci_pfit, width_aci_pfit, cwc_aci_pfit = compute_scores(
        y_pis_aci_pfit, y_test, eta=0.01
    )

    y_enbpi_preds = [y_pred_enbpi_no_pfit, y_pred_enbpi_pfit]
    y_enbpi_pis = [y_pis_enbpi_no_pfit, y_pis_enbpi_pfit]
    coverages_enbpi = [coverage_enbpi_no_pfit, coverage_enbpi_pfit]
    widths_enbpi = [width_enbpi_no_pfit, width_enbpi_pfit]

    y_aci_preds = [y_pred_aci_no_pfit, y_pred_aci_pfit]
    y_aci_pis = [y_pis_aci_no_pfit, y_pis_aci_pfit]
    coverages_aci = [coverage_aci_no_pfit, coverage_aci_pfit]
    widths_aci = [width_aci_no_pfit, width_aci_pfit]

    fig, axs = plt.subplots(
        nrows=2, ncols=1, figsize=(14, 8), sharey="row", sharex="col"
    )
    for i, (ax, w) in enumerate(zip(axs, ["without", "with"])):
        ax.set_ylabel("Hourly demand (GW)")
        ax.plot(y_train[int(-len(y_test) / 2):], lw=2, label="Training data", c="C0")
        ax.plot(y_test, lw=2, label="Test data", c="C1")

        ax.plot(y_test.index, y_enbpi_preds[i], lw=2, c="C2", label="Predictions")
        ax.fill_between(
            y_test.index,
            y_enbpi_pis[i][:, 0, 0],
            y_enbpi_pis[i][:, 1, 0],
            color="C2",
            alpha=0.2,
            label="Prediction intervals",
        )
        title = f"EnbPI, {w} update of residuals. "
        title += (
            f"Coverage:{coverages_enbpi[i]:.3f} and " f"Width:{widths_enbpi[i]:.3f}"
        )
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    io_helper.save_plot(f"{filename}1_{N_POINTS_TEMP}.png")
    plt.show()

    fig, axs = plt.subplots(
        nrows=2, ncols=1, figsize=(14, 8), sharey="row", sharex="col"
    )
    for i, (ax, w) in enumerate(zip(axs, ["without", "with"])):
        ax.set_ylabel("Hourly demand (GW)")
        ax.plot(y_train[int(-len(y_test) / 2):], lw=2, label="Training data", c="C0")
        ax.plot(y_test, lw=2, label="Test data", c="C1")

        ax.plot(y_test.index, y_aci_preds[i], lw=2, c="C2", label="Predictions")
        ax.fill_between(
            y_test.index,
            y_aci_pis[i][:, 0, 0],
            y_aci_pis[i][:, 1, 0],
            color="C2",
            alpha=0.2,
            label="Prediction intervals",
        )
        title = f"ACI, {w} update of residuals. "
        title += f"Coverage:{coverages_aci[i]:.3f} and Width:{widths_aci[i]:.3f}"
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    io_helper.save_plot(f"{filename}2_{N_POINTS_TEMP}.png")
    plt.show()


def plot_prediction_intervals_all_quantiles(
    y_train, y_test, y_pred, y_pis, filename="prediction_intervals_all_quantiles", io_helper: IO_Helper = None,
):
    coverage, width, cwc = compute_scores(y_pis, y_test, eta=10)

    fig, axs = plt.subplots(
        nrows=2, ncols=1, figsize=(14, 8), sharey="row", sharex="col"
    )
    for i, (ax, w) in enumerate(zip(axs, ["without", "with"])):
        ax.set_ylabel("Hourly demand (GW)")
        ax.plot(y_train[int(-len(y_test) / 2):], lw=2, label="Training data", c="C0")
        ax.plot(y_test, lw=2, label="Test data", c="C1")

        ax.plot(y_test.index, y_pred, lw=2, c="C2", label="Predictions")
        ax.fill_between(
            y_test.index,
            y_pis[:, 0],
            y_pis[i][:, 1],
            color="C2",
            alpha=0.2,
            label="Prediction intervals",
        )
        title = f"EnbPI, {w} update of residuals. "
        title += f"Coverage:{coverage:.3f} and " f"Width:{width:.3f}"
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    io_helper.save_plot(f"{filename}1_{N_POINTS_TEMP}.png")
    plt.show()


def compare_coverages(
    y_test,
    y_pis_aci_no_pfit,
    y_pis_aci_pfit,
    y_pis_enbpi_no_pfit,
    y_pis_enbpi_pfit,
    filename="coverages",
    io_helper: IO_Helper = None,
):
    """
    compare coverages obtained by MAPIE with and without update of the residuals on a 24-hour rolling
    window of prediction intervals.
    """
    rolling_coverage_aci_pfit, rolling_coverage_aci_no_pfit = [], []
    rolling_coverage_enbpi_pfit, rolling_coverage_enbpi_no_pfit = [], []

    window = 24

    for i in range(window, len(y_test), 1):
        rolling_coverage_aci_no_pfit.append(
            regression_coverage_score(
                y_test[i - window: i],
                y_pis_aci_no_pfit[i - window: i, 0, 0],
                y_pis_aci_no_pfit[i - window: i, 1, 0],
            )
        )
        rolling_coverage_aci_pfit.append(
            regression_coverage_score(
                y_test[i - window: i],
                y_pis_aci_pfit[i - window: i, 0, 0],
                y_pis_aci_pfit[i - window: i, 1, 0],
            )
        )

        rolling_coverage_enbpi_no_pfit.append(
            regression_coverage_score(
                y_test[i - window: i],
                y_pis_enbpi_no_pfit[i - window: i, 0, 0],
                y_pis_enbpi_no_pfit[i - window: i, 1, 0],
            )
        )
        rolling_coverage_enbpi_pfit.append(
            regression_coverage_score(
                y_test[i - window: i],
                y_pis_enbpi_pfit[i - window: i, 0, 0],
                y_pis_enbpi_pfit[i - window: i, 1, 0],
            )
        )

    plt.figure(figsize=(10, 5))
    plt.ylabel(f"Rolling coverage [{window} hours]")

    plt.plot(
        y_test[window:].index,
        rolling_coverage_aci_no_pfit,
        label="ACI Without update of residuals (NPfit)",
        linestyle="--",
        color="r",
        alpha=0.5,
    )
    plt.plot(
        y_test[window:].index,
        rolling_coverage_aci_pfit,
        label="ACI With update of residuals (Pfit)",
        linestyle="-",
        color="r",
        alpha=0.5,
    )

    plt.plot(
        y_test[window:].index,
        rolling_coverage_enbpi_no_pfit,
        label="ENBPI Without update of residuals (NPfit)",
        linestyle="--",
        color="b",
        alpha=0.5,
    )
    plt.plot(
        y_test[window:].index,
        rolling_coverage_enbpi_pfit,
        label="ENBPI With update of residuals (Pfit)",
        linestyle="-",
        color="b",
        alpha=0.5,
    )

    plt.legend()
    io_helper.save_plot(f"{filename}_{N_POINTS_TEMP}.png")
    plt.show()


if __name__ == "__main__":
    main(IO_HELPER)
