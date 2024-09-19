from itertools import chain

import numpy as np
from mapie.subsample import BlockBootstrap
from matplotlib import pyplot as plt

from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from uncertainty_toolbox import get_all_accuracy_metrics

from helpers import get_data
from mapie_plot_ts_tutorial import (
    train_base_model,
    estimate_pred_interals_no_pfit_enbpi,
)
from quantile_regression import estimate_quantiles

# todo: potential for bugs!
N_POINTS_TEMP = 100  # per group

PLOT_DATA = False

GET_DATA_FUNC = lambda: get_data(N_POINTS_TEMP, return_full_data=True)

model_params_choices = {"max_depth": randint(2, 30), "n_estimators": randint(10, 100)}
TRAIN_BASE_MODEL_FUNC = lambda X_train, y_train: train_base_model(
    RandomForestRegressor,
    model_params_choices=model_params_choices,
    X_train=X_train,
    y_train=y_train,
    load_trained_model=True,
    cv_n_iter=10,
)

NATIVE_METHODS = {
    "QR": lambda X_train, y_train, X_test: estimate_quantiles(
        X_train, y_train, X_test, ci_alpha=0.1
    ),
}


def estimate_quantiles_cp(model, X_train, y_train, X_test):
    random_state = 59
    alpha = 0.05
    cv_mapie_ts = BlockBootstrap(
        n_resamplings=10, n_blocks=10, overlapping=False, random_state=random_state
    )
    y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit = estimate_pred_interals_no_pfit_enbpi(
        model, cv_mapie_ts, alpha, X_test, X_train, y_train, skip_base_training=True
    )
    return y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit


POSTHOC_METHODS = {"CP": estimate_quantiles_cp}


METRICS = {
    "all": lambda y_pred, y_pis, y_true: get_all_accuracy_metrics(
        y_pred, y_true.to_numpy().squeeze()
    ),
}


def compare_methods(
    get_data_func,
    train_base_model_func,
    native_methods,
    posthoc_methods,
    metrics,
    should_plot_data=True,
) -> tuple[dict[str, tuple[np.array, np.array]], dict[str, tuple[np.array, np.array]]]:
    """

    :param get_data_func:
    :param train_base_model_func:
    :param native_methods:
    :param posthoc_methods:
    :param metrics:
    :param should_plot_data:
    :return: native_results, posthoc_results
    """
    print("loading data")
    X_train, X_test, y_train, y_test, X, y = get_data_func()
    print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    if should_plot_data:
        print("plotting data")
        plot_data(X_train, X_test, y_train, y_test)

    print("running native methods")
    native_results = run_uq_methods(X_test, X_train, y_train, native_methods)
    posthoc_results = run_uq_methods(
        X_test, X_train, y_train, posthoc_methods, train_base_model_func
    )

    print("plotting native vs posthoc results")
    plot_intervals(X_train, y_train, X_test, y, native_results, posthoc_results)

    # print("computing metrics")
    # native_metrics, posthoc_metrics = compute_metrics(
    #     metrics, native_results, posthoc_results, y_test
    # )
    return native_results, posthoc_results


def compute_metrics(metrics, native_results, posthoc_results, y_test):
    native_metrics, posthoc_metrics = {}, {}
    for method_name, (y_pred, y_pis) in chain(
        native_results.items(), posthoc_results.items()
    ):
        print(f"\n========= {method_name} =========\n")
        metric_dict = (
            native_metrics if method_name in native_results else posthoc_metrics
        )
        for metric_name, metric_func in metrics.items():
            metric_value = metric_func(y_pred, y_pis, y_test)
            metric_dict[metric_name] = metric_value
            print(f"{metric_name}: {metric_value}")

    # acc_cp, acc_qr = ..., ...  # todo
    # print('accuracy CP vs QR:', acc_cp, acc_qr)
    #
    # crps_cp, crps_qr = ..., ...  # todo
    # print('CRPS CP vs QR:', crps_cp, crps_qr)
    return native_metrics, posthoc_metrics


def run_uq_methods(X_test, X_train, y_train, uq_methods, train_base_model_func=None):
    is_posthoc = train_base_model_func is not None
    if is_posthoc:
        print("training base model")
        model = train_base_model_func(X_train, y_train)

    print(f"running {'posthoc' if is_posthoc else 'native'} methods")
    uq_results = {}
    for method_name, uq_method in uq_methods.items():
        print(f"estimating {method_name} intervals")
        if is_posthoc:
            # noinspection PyUnboundLocalVariable
            y_pred, y_pis = uq_method(model, X_train, y_train, X_test)
        else:
            y_pred, y_pis = uq_method(X_train, y_train, X_test)
        uq_results[method_name] = y_pred, y_pis
    return uq_results


# PLOTTING


def plot_data(
    X_train,
    X_test,
    y_train,
    y_test,
    figsize=(16, 5),
    ylabel="energy data (details TODO)",
):
    """visualize training and test sets"""
    num_train_steps = X_train.shape[0]
    num_test_steps = X_test.shape[0]

    x_plot_train = np.arange(num_train_steps)
    x_plot_test = x_plot_train + num_test_steps

    plt.figure(figsize=figsize)
    plt.plot(x_plot_train, y_train)
    plt.plot(x_plot_test, y_test)
    plt.ylabel(ylabel)
    plt.legend(["Training data", "Test data"])
    plt.show()


def plot_intervals(X_train, y_train, X_test, y, native_results, posthoc_results):
    res_dict = {"native": native_results, "posthoc": posthoc_results}
    for res_type, results in res_dict.items():
        print(f"plotting {res_type} results...")
        for method_name, (y_preds, y_pis) in results.items():
            plot_uq_results(X_train, X_test, y_train, y, y_pis, y_preds, method_name)


def plot_uq_results(X_train, X_test, y_train, y, y_pis, y_preds, method_name):
    num_train_steps = X_train.shape[0]
    num_test_steps = X_test.shape[0]
    num_steps_total = num_train_steps + num_test_steps

    x_plot_train = np.arange(num_train_steps)
    x_plot_test = x_plot_train + num_test_steps
    x_plot_full = np.arange(num_steps_total)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    ax.plot(x_plot_full, y, color="black", linestyle="dashed", label="True mean")
    ax.scatter(
        x_plot_train,
        y_train,
        color="black",
        marker="o",
        alpha=0.8,
        label="training points",
    )
    ax.plot(
        x_plot_test,
        y_preds,
        label=f"mean/median prediction {method_name}",
        color="green",
    )
    ax.fill_between(
        x_plot_test.ravel(),
        y_pis[:, 0],
        y_pis[:, 1],
        color="green",
        alpha=0.2,
        label=rf"{method_name} 95% confidence interval",
    )
    ax.legend()
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title(method_name)
    plt.show()


if __name__ == "__main__":
    compare_methods(
        GET_DATA_FUNC,
        TRAIN_BASE_MODEL_FUNC,
        NATIVE_METHODS,
        POSTHOC_METHODS,
        METRICS,
        should_plot_data=PLOT_DATA,
    )
