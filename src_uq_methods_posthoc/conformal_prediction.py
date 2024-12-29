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

import numpy as np
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from helpers import misc_helpers


def train_conformal_prediction(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_model,
    n_estimators=10,
    bootstrap_n_blocks=10,
    bootstrap_overlapping_blocks=False,
    random_seed=42,
    verbose=1,
):
    cv = BlockBootstrap(
        n_resamplings=n_estimators,
        n_blocks=bootstrap_n_blocks,
        overlapping=bootstrap_overlapping_blocks,
        random_state=random_seed,
    )
    model = MapieTimeSeriesRegressor(
        base_model, method="enbpi", cv=cv, agg_function='mean', n_jobs=-1, verbose=verbose,
    )
    model = model.fit(X_train, y_train)
    return model


def predict_with_conformal_prediction(model, X_pred: np.ndarray, quantiles: list):
    alpha = misc_helpers.pis_from_quantiles(quantiles)
    try:
        alpha = list(alpha)
    except TypeError:
        pass

    # y_pis is of shape (n_samples, 2, n_alpha), with:
    # [:, 0, alpha]: Lower bound of the prediction interval corresp. to confidence level alpha.
    # [:, 1, alpha]: Upper bound of the prediction interval corresp. to confidence level alpha.
    y_pred, y_pis = model.predict(
        X_pred, alpha=alpha, ensemble=False, optimize_beta=False, allow_infinite_bounds=True,
    )
    y_quantiles = misc_helpers.quantiles_from_pis(y_pis)  # (n_samples, 2 * n_intervals)
    if 0.5 in quantiles:
        num_quantiles = y_quantiles.shape[-1]
        ind = num_quantiles // 2
        y_quantiles = np.insert(y_quantiles, ind, y_pred, axis=1)
    y_std = misc_helpers.stds_from_quantiles(y_quantiles)
    return y_pred, y_quantiles, y_std
