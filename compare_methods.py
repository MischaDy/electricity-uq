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


def extimate_quantiles_cp(model, X_test, X_train, y_train):
    random_state = 59
    alpha = 0.05
    cv_mapie_ts = BlockBootstrap(
        n_resamplings=10, n_blocks=10, overlapping=False, random_state=random_state
    )
    y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit = estimate_pred_interals_no_pfit_enbpi(
        model, cv_mapie_ts, alpha, X_test, X_train, y_train, skip_base_training=True
    )
    return y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit


POSTHOC_METHODS = {"CP": extimate_quantiles_cp}


METRICS = {
    "all": lambda y_pred, y_pis, y_true: get_all_accuracy_metrics(
        y_pred, y_true.to_numpy().squeeze()
    ),
}


def main(
    get_data_func,
    train_base_model_func,
    native_methods,
    posthoc_methods,
    metrics,
    should_plot_data=True,
):
    print("loading data")
    X_train, X_test, y_train, y_test, X, y = get_data_func()
    print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    if should_plot_data:
        print("plotting data")
        plot_data(X_train, X_test, y_train, y_test)

    print("running native methods")
    native_results = {}
    for name, native_method in native_methods.items():
        print(f"estimating {name} intervals")
        y_pred_native, y_pis_native = native_method(X_train, y_train, X_test)
        native_results[name] = y_pred_native, y_pis_native

    print("training base model")
    model = train_base_model_func(X_train, y_train)

    print("running posthoc methods")
    posthoc_results = {}
    for name, posthoc_method in posthoc_methods.items():
        print(f"estimating {name} intervals")
        y_pred_posthoc, y_pis_posthoc = posthoc_method(model, X_train, y_train, X_test)
        posthoc_results[name] = y_pred_posthoc, y_pis_posthoc

    print("plotting native vs posthoc results")
    plot_intervals(
        X, y, X_train, y_train, X_test, y_test, native_results, posthoc_results
    )

    print("computing metrics")

    native_metrics = {}
    for name, native_result in native_results.items():
        print(f"========= {name} =========\n")
        y_pred_native, y_pis_native = native_result
        for metric_name, metric_func in metrics.items():
            metric_value = metric_func(y_pred_native, y_pis_native, y_test)
            native_metrics[metric_name] = metric_value
            print(f"{metric_name}: {metric_value}")

    posthoc_metrics = {}
    for name, posthoc_result in posthoc_results.items():
        print(f"========= {name} =========\n")
        y_pred_posthoc, y_pis_posthoc = posthoc_result
        for metric_name, metric_func in metrics.items():
            metric_value = metric_func(y_pred_posthoc, y_pis_posthoc, y_test)
            posthoc_metrics[metric_name] = metric_value
            print(f"{metric_name}: {metric_value}")

    # acc_cp, acc_qr = ..., ...  # todo
    # print('accuracy CP vs QR:', acc_cp, acc_qr)
    #
    # crps_cp, crps_qr = ..., ...  # todo
    # print('CRPS CP vs QR:', crps_cp, crps_qr)


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


def plot_intervals(
    X, y, X_train, y_train, X_test, y_test, native_results, posthoc_results
):
    num_train_steps = X_train.shape[0]
    num_test_steps = X_test.shape[0]
    num_steps_total = num_train_steps + num_test_steps
    assert X.shape[0] == num_steps_total

    x_plot_train = np.arange(num_train_steps)
    x_plot_test = x_plot_train + num_test_steps
    x_plot_full = np.arange(num_steps_total)

    # todo: what to plot?
    fig, axs = plt.subplots(
        nrows=2, ncols=1, figsize=(14, 8), sharey="row", sharex="col"
    )
    for i, (ax, y_pred, y_pis, label) in enumerate(zip(axs, y_preds, y_pis, labels)):
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
            x_plot_test, y_pred, label=f"mean/median prediction {label}", color="green"
        )
        ax.fill_between(
            x_plot_test.ravel(),
            y_pis[:, 0],
            y_pis[:, 1],
            color="green",
            alpha=0.2,
            label=rf"{label} 95% confidence interval",
        )

        ax.legend()
        ax.set_xlabel("data")
        ax.set_ylabel("target")
        ax.set_title(label)
    plt.show()


if __name__ == "__main__":
    main(
        GET_DATA_FUNC,
        TRAIN_BASE_MODEL_FUNC,
        NATIVE_METHODS,
        POSTHOC_METHODS,
        should_plot_data=True,
    )
