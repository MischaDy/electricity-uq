import torch
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from matplotlib import pyplot as plt

from more_itertools import collapse
from tqdm import tqdm
from uncertainty_toolbox import nll_gaussian

from helpers import numpy_to_tensor, tensor_to_numpy, get_train_loader, get_data, standardize, df_to_numpy, \
    set_dtype_float, plot_data

QUANTILES = [
    0.05,
    0.25,
    0.75,
    0.95,
]
TO_STANDARDIZE = "xy"
N_POINTS_PER_GROUP = 200
N_ITER = 200

torch.set_default_dtype(torch.float32)


class MeanVarNN(nn.Module):
    def __init__(
        self,
        dim_in,
        num_hidden_layers=2,
        hidden_layer_size=50,
        activation=torch.nn.LeakyReLU,
    ):
        super().__init__()
        layers = collapse([
            torch.nn.Linear(dim_in, hidden_layer_size),
            activation(),
            [[torch.nn.Linear(hidden_layer_size, hidden_layer_size),
              activation()]
             for _ in range(num_hidden_layers)],
            torch.nn.Linear(hidden_layer_size, 2),
        ])
        self.first_layer_stack = torch.nn.Sequential(*layers)

    def forward(self, x):
        # todo: make tensor if isnt
        x = self.first_layer_stack(x)
        x = self.output_activation(x)
        return x

    @staticmethod
    def output_activation(x, eps=0.01) -> tuple[torch.Tensor, torch.Tensor]:
        """

        :param eps:
        :param x: tensor of shape (n_samples, 2), where the dimension 1 are means and dimension 2 are variances
        :return:
        """
        mean, var = x[:, 0], x[:, 1]
        var = F.relu(var)
        var = torch.clamp_min(var, min=eps)
        return mean, var


def train_mean_var_nn(
    X,
    y,
    n_iter=200,
    batch_size=20,
    random_state=711,
    val_frac=0.1,
    lr=0.1,
    lr_patience=5,
    lr_reduction_factor=0.5,
    verbose=True,
):
    torch.manual_seed(random_state)

    try:
        X_train, y_train = map(numpy_to_tensor, (X, y))
    except TypeError:
        raise TypeError(f'Unknown label type: {X.dtype} (X) or {y.dtype} (y)')

    n_samples = X_train.shape[0]
    val_size = max(1, round(val_frac * n_samples))
    train_size = max(1, n_samples - val_size)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:train_size], y_train[:train_size]

    assert X_train.shape[0] > 0 and X_val.shape[0] > 0

    dim_in, dim_out = X_train.shape[-1], y_train.shape[-1]

    model = MeanVarNN(
        dim_in,
        num_hidden_layers=2,
        hidden_layer_size=50,
    )

    train_loader = get_train_loader(X_train, y_train, batch_size)

    criterion = nn.GaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, patience=lr_patience, factor=lr_reduction_factor
    )

    train_losses, val_losses = [], []
    # iterable = tqdm(range(n_iter)) if verbose else range(n_iter)
    iterable = range(n_iter)
    for _ in iterable:
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            y_pred_mean, y_pred_var = model(X)
            loss = criterion(y_pred_mean, y, y_pred_var)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = _nll_loss_np(model(X_val), y_val)
            train_loss = _nll_loss_np(model(X_train[:val_size]), y_train[:val_size])
        scheduler.step(val_loss)
        val_losses.append(val_loss)
        train_losses.append(train_loss)
    if verbose:
        loss_skip = 0
        plot_training_progress(train_losses[loss_skip:], val_losses[loss_skip:])

    model.eval()
    return model


def plot_training_progress(train_losses, test_losses):
    fig, ax = plt.subplots()
    ax.semilogy(train_losses, label="train")
    ax.semilogy(test_losses, label="val")
    ax.legend()
    plt.show()


def _nll_loss_np(y_pred, y_test):
    y_pred_mean, y_pred_var = y_pred
    tensors = y_pred_mean, np.sqrt(y_pred_var), y_test
    arrs = map(lambda t: t.numpy(force=True).squeeze(), tensors)
    return nll_gaussian(*arrs)


# def predict(model, X, as_np=True):
#     X = numpy_to_tensor(X)
#     with torch.no_grad():
#         tensor_pair = model(X)
#     # tensor_pair = map(lambda t: t.reshape(-1, 1) if is_y_2d_ else t.squeeze(), tensor_pair)
#     if as_np:
#         return tuple(map(lambda t: tensor_to_numpy(t).squeeze(), tensor_pair))
#     return tensor_pair


def run_mean_var_nn(X_train, y_train, X_test, quantiles):
    X_train, y_train, X_test = map(numpy_to_tensor, (X_train, y_train, X_test))
    mean_var_nn = train_mean_var_nn(X_train, y_train, n_iter=N_ITER)
    # plot_post_training_perf(mean_var_nn, X_train, y_train)
    with torch.no_grad():
        y_pred, y_var = mean_var_nn(X_test)
    y_pred, y_var = map(tensor_to_numpy, (y_pred, y_var))
    y_std = np.sqrt(y_var)
    y_quantiles = quantiles_gaussian(quantiles, y_pred, y_std)
    return y_pred, y_quantiles, y_std


def quantiles_gaussian(quantiles, y_pred, y_std):
    # todo: does this work for multi-dim outputs?
    return np.array([norm.ppf(quantiles, loc=mean, scale=std)
                     for mean, std in zip(y_pred, y_std)])


# def plot_post_training_perf(base_model, X_train, y_train):
#     y_pred_mean, y_pred_var = base_model.predict(X_train)
#
#     num_train_steps = X_train.shape[0]
#     x_plot_train = np.arange(num_train_steps)
#
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
#     ax.plot(x_plot_train, y_train, label='y_train', linestyle="dashed", color="black")
#     ax.plot(
#         x_plot_train,
#         y_pred_mean,
#         label=f"base model prediction",
#         color="green",
#     )
#     ax.legend()
#     ax.set_xlabel("data")
#     ax.set_ylabel("target")
#     plt.show()


def plot_uq_result(
    X_train,
    X_test,
    y_train,
    y_test,
    y_preds,
    y_quantiles,
    y_std,
    quantiles,
):
    num_train_steps, num_test_steps = X_train.shape[0], X_test.shape[0]

    x_plot_train = np.arange(num_train_steps)
    x_plot_full = np.arange(num_train_steps + num_test_steps)
    x_plot_test = np.arange(num_train_steps, num_train_steps + num_test_steps)
    x_plot_uq = x_plot_full

    drawing_std = y_quantiles is not None
    if drawing_std:
        ci_low, ci_high = (
            y_quantiles[:, 0],
            y_quantiles[:, -1],
        )
        drawn_quantile = round(max(quantiles) - min(quantiles), 2)
    else:
        ci_low, ci_high = y_preds - y_std / 2, y_preds + y_std / 2

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    ax.plot(x_plot_train, y_train, label='y_train', linestyle="dashed", color="black")
    ax.plot(x_plot_test, y_test, label='y_test', linestyle="dashed", color="blue")
    ax.plot(
        x_plot_uq,
        y_preds,
        label=f"mean/median prediction",  # todo: mean or median?
        color="green",
    )
    # noinspection PyUnboundLocalVariable
    label = rf"{f'{100*drawn_quantile}% CI' if drawing_std else '1 std'}"
    ax.fill_between(
        x_plot_uq.ravel(),
        ci_low,
        ci_high,
        color="green",
        alpha=0.2,
        label=label,
    )
    ax.legend()
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    plt.show()


def make_2d(arr):
    return arr.reshape(-1, 1)


# todo
def get_clean_data2(_n_points_per_group=100):
    X_train, X_test, y_train, y_test, X, y = get_data(_n_points_per_group, return_full_data=True)
    X_train, X_test, X = _standardize_or_to_array("x", X_train, X_test, X)
    y_train, y_test, y = _standardize_or_to_array("y", y_train, y_test, y)
    return X_train, X_test, y_train, y_test, X, y


def get_clean_data(_n_points_per_group=100):
    n_points = 2 * _n_points_per_group
    X = np.arange(n_points) + 1
    y = 20 + 10 * np.sin(10 * X * 2*np.pi/n_points) + np.random.randn(n_points)

    X, y = map(make_2d, (X, y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )
    X_train, X_test, y_train, y_test, X, y = set_dtype_float(X_train, X_test, y_train, y_test, X, y)

    plot_data(X_train, X_test, y_train, y_test)

    X_train, X_test, X = _standardize_or_to_array("x", X_train, X_test, X)
    y_train, y_test, y = _standardize_or_to_array("y", y_train, y_test, y)
    return X_train, X_test, y_train, y_test, X, y


def _standardize_or_to_array(variable, *dfs):
    if variable in TO_STANDARDIZE:
        return standardize(False, *dfs)
    return map(df_to_numpy, dfs)


def main():
    print("loading data...")
    X_train, X_test, y_train, y_test, X, y = get_clean_data(N_POINTS_PER_GROUP)
    print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    print("running method...")
    X_uq = np.row_stack((X_train, X_test))

    y_pred, y_quantiles, y_std = run_mean_var_nn(X_train, y_train, X_uq, QUANTILES)

    plot_uq_result(
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_quantiles,
        y_std,
        QUANTILES
    )


if __name__ == '__main__':
    main()
