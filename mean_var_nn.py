import torch
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from matplotlib import pyplot as plt

from more_itertools import collapse
from tqdm import tqdm
from uncertainty_toolbox import nll_gaussian

from helpers import numpy_to_tensor, tensor_to_numpy, get_train_loader, get_data, standardize, df_to_numpy, \
    set_dtype_float

QUANTILES = [
    0.05,
    0.25,
    0.75,
    0.95,
]

TO_STANDARDIZE = "xy"
PLOT_DATA = False

N_POINTS_PER_GROUP = 800

N_ITER = 500
LR = 1e-2
LR_PATIENCE = 30
REGULARIZATION = 0  # 1e-2
USE_SCHEDULER = True
WARMUP_PERIOD = 100
FROZEN_VAR_VALUE = 0.2


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
            nn.Linear(dim_in, hidden_layer_size),
            activation(),
            [[nn.Linear(hidden_layer_size, hidden_layer_size),
              activation()]
             for _ in range(num_hidden_layers)],
        ])
        self.first_layer_stack = nn.Sequential(*layers)
        self.last_layer_mean = nn.Linear(hidden_layer_size, 1)
        self.last_layer_var = nn.Linear(hidden_layer_size, 1)
        self._frozen_var = None

    def forward(self, x):
        # todo: make tensor if isn't?
        x = self.first_layer_stack(x)
        mean = self.last_layer_mean(x)
        var = torch.exp(self.last_layer_var(x))
        if self._frozen_var is not None:
            var = torch.full(var.shape, self._frozen_var)
        return mean, var

    @staticmethod
    def output_activation(x) -> tuple[torch.Tensor, torch.Tensor]:
        mean, var = x[:, 0], x[:, 1]
        var = torch.exp(var)
        return mean, var

    def freeze_variance(self, value: float):
        assert value > 0
        self.last_layer_var.requires_grad_(False)
        self._frozen_var = value

    def unfreeze_variance(self):
        self.last_layer_var.requires_grad_(True)
        self._frozen_var = None


def train_mean_var_nn(
    X,
    y,
    model: MeanVarNN = None,
    n_iter=200,
    batch_size=20,
    random_state=711,
    val_frac=0.1,
    lr=0.1,
    lr_patience=5,
    lr_reduction_factor=0.5,
    weight_decay=0.0,
    train_var=True,
    plot_skip_losses=10,
    verbose=True,
    frozen_var_value=0.5
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

    if model is None:
        model = MeanVarNN(
            dim_in,
            num_hidden_layers=2,
            hidden_layer_size=50,
        )

    train_loader = get_train_loader(X_train, y_train, batch_size)

    criterion = nn.GaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_reduction_factor)

    if train_var:
        model.freeze_variance(frozen_var_value)
    else:
        model.unfreeze_variance()

    train_losses, val_losses = [], []
    iterable = np.arange(n_iter) + 1
    if verbose:
        iterable = tqdm(iterable)
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
        if USE_SCHEDULER:
            scheduler.step(val_loss)
        val_losses.append(val_loss)
        train_losses.append(train_loss)
    if verbose:
        plot_training_progress(train_losses[plot_skip_losses:], val_losses[plot_skip_losses:])

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


def run_mean_var_nn(X_train, y_train, X_test, quantiles):
    X_train, y_train, X_test = map(numpy_to_tensor, (X_train, y_train, X_test))
    common_params = {
        "lr": LR,
        "lr_patience": LR_PATIENCE,
        "weight_decay": REGULARIZATION,
    }
    mean_var_nn = None
    if WARMUP_PERIOD > 0:
        print('running warmup...')
        mean_var_nn = train_mean_var_nn(
            X_train, y_train, n_iter=WARMUP_PERIOD, train_var=False, frozen_var_value=FROZEN_VAR_VALUE,
            **common_params
        )
    mean_var_nn = train_mean_var_nn(
        X_train, y_train, model=mean_var_nn, n_iter=N_ITER, train_var=True,
        **common_params
    )
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
    label = rf"{f'{100 * drawn_quantile}% CI' if drawing_std else '1 std'}"
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
def get_clean_data(_n_points_per_group=100):
    X_train, X_test, y_train, y_test, X, y = get_data(_n_points_per_group, return_full_data=True)
    X_train, X_test, X = _standardize_or_to_array("x", X_train, X_test, X)
    y_train, y_test, y = _standardize_or_to_array("y", y_train, y_test, y)

    if PLOT_DATA:
        my_plot_data(X, y)

    return X_train, X_test, y_train, y_test, X, y


def get_clean_data2(_n_points_per_group=100):
    n_points = 2 * _n_points_per_group
    X_base = np.arange(n_points) + 1

    num_cycles = 10
    period = num_cycles * 2 * np.pi / n_points

    X = 20 + 10 * np.sin(period * X_base) + np.random.randn(n_points)

    shift = n_points // num_cycles  # round(10 * period)
    y = X[shift:]
    X = X[:len(y)]

    X, y = map(make_2d, (X, y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )
    X_train, X_test, y_train, y_test, X, y = set_dtype_float(X_train, X_test, y_train, y_test, X, y)

    if PLOT_DATA:
        my_plot_data(X, y)

    # plot_data(X_train, X_test, y_train, y_test)

    X_train, X_test, X = _standardize_or_to_array("x", X_train, X_test, X)
    y_train, y_test, y = _standardize_or_to_array("y", y_train, y_test, y)
    return X_train, X_test, y_train, y_test, X, y


def my_plot_data(X, y):
    x_plot = np.arange(X.shape[0])
    if X.shape[-1] == 1:
        plt.plot(x_plot, X, label='X')
    plt.plot(x_plot, y, label='y')
    plt.legend()
    plt.show()


def _standardize_or_to_array(variable, *dfs):
    if variable in TO_STANDARDIZE:
        return standardize(False, *dfs)
    return map(df_to_numpy, dfs)


def main():
    torch.set_default_dtype(torch.float32)
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
