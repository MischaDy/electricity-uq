#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import QuantileRegressor

from helpers import get_data

# source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html


# todo: potential for bugs!
N_POINTS_TEMP = 100  # per group


def main():
    X_train, X_test, y_train, y_test, X, y = get_data(N_POINTS_TEMP, return_full_data=True)

    y_preds, y_pis = estimate_quantiles(X, y, X_train, y_train, ci_width=0.1)

    plot(X, y, X_train, y_train, y_preds, y_pis)


def estimate_quantiles(X, y, X_train, y_train, ci_width=0.1):
    quant_min = ci_width/2
    quant_median = 0.5
    quant_max = 1 - quant_min

    predictions = {}
    for quantile in [quant_min, quant_median, quant_max]:
        qr = QuantileRegressor(quantile=quantile, alpha=0)
        y_pred = qr.fit(X_train, y_train.values.ravel()).predict(X)
        predictions[quantile] = y_pred
    y_preds = predictions[quant_median]
    y_pis = np.stack([predictions[quant_min], predictions[quant_max]]).reshape((-1, 2))  # (n_samples, 2)
    return y_preds, y_pis


def plot(X, y, X_train, y_train, y_preds, y_pis):
    num_train_steps = X_train.shape[0]
    num_steps_total = X.shape[0]
    x_plot_train = np.arange(num_train_steps)
    x_plot_full = np.arange(num_steps_total)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_plot_full, y, color="black", linestyle="dashed", label="True mean")

    plt.scatter(
        x_plot_train,
        y_train,
        color="black",
        marker="o",
        alpha=0.8,
        label="training points",
    )

    ax.plot(x_plot_full, y_preds, label="QR median prediction", color='green')
    ax.fill_between(
        x_plot_full.ravel(),
        y_pis[:, 0],
        y_pis[:, 1],
        color='green',
        alpha=0.2,
        label=r"QR 95% confidence interval",
    )

    plt.legend()
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("QR")
    plt.show()


def plot_data(X_train, X_test, y_train, y_test):
    """visualize training and test sets"""
    # todo: extract
    num_train_steps = X_train.shape[0]
    num_test_steps = X_test.shape[0]

    x_plot_train = np.arange(num_train_steps)
    x_plot_test = x_plot_train + num_test_steps

    plt.figure(figsize=(16, 5))
    plt.plot(x_plot_train, y_train)
    plt.plot(x_plot_test, y_test)
    plt.ylabel("energy data (details TODO)")
    plt.legend(["Training data", "Test data"])
    plt.show()


if __name__ == '__main__':
    main()
