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


def main():
    print("loading data")
    X_train, X_test, y_train, y_test, X, y = get_data(
        N_POINTS_TEMP, return_full_data=True
    )
    print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # print('plotting data')
    # plot_data(X_train, X_test, y_train, y_test)

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
        load_trained_model=True,
        cv_n_iter=10,
    )

    print("estimating CP intervals")
    random_state = 59
    alpha = 0.05
    cv_mapie_ts = BlockBootstrap(
        n_resamplings=10, n_blocks=10, overlapping=False, random_state=random_state
    )
    y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit = estimate_pred_interals_no_pfit_enbpi(
        model, cv_mapie_ts, alpha, X_test, X_train, y_train, skip_base_training=True
    )

    print("estimating QR intervals")
    y_pred_qr, y_pis_qr = estimate_quantiles(X_train, y_train, X_test, ci_alpha=0.1)

    print("plotting CP vs QR")
    y_preds_both = [y_pred_enbpi_no_pfit, y_pred_qr]
    y_pis_both = [y_pis_enbpi_no_pfit, y_pis_qr]
    labels_both = ["CP", "QR"]
    plot_intervals(
        X, y, X_train, y_train, X_test, y_test, y_preds_both, y_pis_both, labels_both
    )

    print("computing metrics")

    print("========= CP =========\n")
    y_test_arr = y_test.to_numpy().squeeze()
    acc_metrics_cp = get_all_accuracy_metrics(y_pred_enbpi_no_pfit, y_test_arr)
    print(acc_metrics_cp)

    print("========= QR =========\n")
    acc_metrics_qr = get_all_accuracy_metrics(y_pred_qr, y_test_arr)
    print(acc_metrics_qr)

    # acc_cp, acc_qr = ..., ...  # todo
    # print('accuracy CP vs QR:', acc_cp, acc_qr)
    #
    # crps_cp, crps_qr = ..., ...  # todo
    # print('CRPS CP vs QR:', crps_cp, crps_qr)


# PLOTTING


def plot_data(X_train, X_test, y_train, y_test):
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
    plt.show()


def plot_intervals(
    X, y, X_train, y_train, X_test, y_test, y_preds_both, y_pis_both, labels_both
):
    num_train_steps = X_train.shape[0]
    num_test_steps = X_test.shape[0]
    num_steps_total = num_train_steps + num_test_steps
    assert X.shape[0] == num_steps_total

    x_plot_train = np.arange(num_train_steps)
    x_plot_test = x_plot_train + num_test_steps
    x_plot_full = np.arange(num_steps_total)

    fig, axs = plt.subplots(
        nrows=2, ncols=1, figsize=(14, 8), sharey="row", sharex="col"
    )
    for i, (ax, y_pred, y_pis, label) in enumerate(
        zip(axs, y_preds_both, y_pis_both, labels_both)
    ):
        ax.plot(x_plot_full, y, color="black", linestyle="dashed", label="True mean")

        ax.scatter(
            x_plot_train,
            y_train,
            color="black",
            marker="o",
            alpha=0.8,
            label="training points",
        )

        ax.plot(x_plot_test, y_pred, label=f"median prediction {label}", color="green")
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
    main()
