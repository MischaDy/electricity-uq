import torch
from scipy.stats import norm
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from matplotlib import pyplot as plt

from more_itertools import collapse
from tqdm import tqdm
from uncertainty_toolbox import nll_gaussian

from helpers import get_train_loader, get_data, standardize, \
    tensors_to_np_arrays, dfs_to_np_arrays, \
    np_arrays_to_tensors, tensors_to_device, train_val_split

QUANTILES = [0.05, 0.25, 0.75, 0.95]

TO_STANDARDIZE = "xy"
PLOT_DATA = False

N_POINTS_PER_GROUP = 800

N_ITER = 100
LR = 1e-4
REGULARIZATION = 0  # 1e-2
WARMUP_PERIOD = 50
FROZEN_VAR_VALUE = 0.1
USE_SCHEDULER = False
LR_PATIENCE = 30


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
        # self.last_layer_var.requires_grad_(False)
        self._frozen_var = value

    def unfreeze_variance(self):
        # self.last_layer_var.requires_grad_(True)
        self._frozen_var = None


def train_mean_var_nn(
    X,
    y,
    model: MeanVarNN = None,
    n_iter=200,
    batch_size=20,
    random_seed=42,
    val_frac=0.1,
    lr=0.1,
    lr_patience=30,
    lr_reduction_factor=0.5,
    weight_decay=0.0,
    train_var=True,
    frozen_var_value=0.5,
    show_progress=True,
    do_plot_losses=True,
    plot_skip_losses=10,
    use_scheduler=True,
):
    torch.manual_seed(random_seed)

    try:
        X_train, y_train = np_arrays_to_tensors(X, y)
    except TypeError:
        raise TypeError(f'Unknown label type: {X.dtype} (X) or {y.dtype} (y)')
    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_frac)

    assert X_train.shape[0] > 0 and X_val.shape[0] > 0

    dim_in, dim_out = X_train.shape[-1], y_train.shape[-1]

    if model is None:
        model = MeanVarNN(
            dim_in,
            num_hidden_layers=2,
            hidden_layer_size=50,
        )

    train_loader = get_train_loader(X_train, y_train, batch_size)

    if train_var:
        model.unfreeze_variance()
        criterion = nn.GaussianNLLLoss()
    else:
        model.freeze_variance(frozen_var_value)
        _mse_loss = nn.MSELoss()
        criterion = lambda input_, target, var: _mse_loss(input_, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_reduction_factor)

    train_losses, val_losses = [], []
    temp_vars = []
    iterable = np.arange(n_iter) + 1
    if show_progress:
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
            train_loss = _nll_loss_np(model(X_train), y_train)
            temp_vars.append(model(X_train)[1].mean())
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        if use_scheduler:
            scheduler.step(val_loss)
    if do_plot_losses:
        for loss_type, losses in {'train_losses': train_losses, 'val_losses': val_losses}.items():
            print(loss_type, train_losses[:5], min(losses), max(losses), any(np.isnan(losses)))
        plot_losses(train_losses[plot_skip_losses:], val_losses[plot_skip_losses:])

    model.eval()
    return model


def plot_losses(train_losses, val_losses):
    def has_neg(losses):
        return any(map(lambda x: x < 0, losses))

    fig, ax = plt.subplots()
    plt_func = ax.plot if has_neg(train_losses) or has_neg(val_losses) else ax.semilogy
    plt_func(train_losses, label="train loss")
    plt_func(val_losses, label="validation loss")
    ax.legend()
    plt.show()


def _nll_loss_np(y_pred, y_test):
    # todo: use nn.NLLLoss instead!
    y_pred_mean, y_pred_var = y_pred
    tensors = y_pred_mean, np.sqrt(y_pred_var), y_test
    arrs = map(lambda t: t.numpy(force=True).squeeze(), tensors)
    return nll_gaussian(*arrs)


def run_mean_var_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    quantiles,
    n_iter=100,
    lr=1e-4,
    lr_patience=5,
    regularization=0,
    warmup_period=10,
    frozen_var_value=0.1,
    do_plot_losses=False,
    use_scheduler=True,
):

    X_train, y_train, X_test = np_arrays_to_tensors(X_train, y_train, X_test)
    X_train, y_train, X_test = tensors_to_device(X_train, y_train, X_test)
    common_params = {
        "lr": lr,
        "lr_patience": lr_patience,
        "weight_decay": regularization,
        "use_scheduler": use_scheduler,
    }
    mean_var_nn = None
    if warmup_period > 0:
        print('running warmup...')
        mean_var_nn = train_mean_var_nn(
            X_train, y_train, n_iter=warmup_period, train_var=False, frozen_var_value=frozen_var_value,
            do_plot_losses=False,
            **common_params
        )
    mean_var_nn = train_mean_var_nn(
        X_train, y_train, model=mean_var_nn, n_iter=n_iter, train_var=True, do_plot_losses=do_plot_losses,
        **common_params
    )
    # plot_post_training_perf(mean_var_nn, X_train, y_train)
    with torch.no_grad():
        y_pred, y_var = mean_var_nn(X_test)
    y_pred, y_var = tensors_to_np_arrays(y_pred, y_var)
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
        ci_low, ci_high = y_preds - y_std, y_preds + y_std

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


def get_clean_data(n_points_per_group, to_standardize, do_plot_data=True):
    X_train, X_test, y_train, y_test, X, y = get_data(n_points_per_group, return_full_data=True)
    X_train, X_test, X = _standardize_or_to_array("x", to_standardize, X_train, X_test, X)
    y_train, y_test, y = _standardize_or_to_array("y", to_standardize, y_train, y_test, y)
    if do_plot_data:
        my_plot_data(X, y)
    return X_train, X_test, y_train, y_test, X, y


def my_plot_data(X, y):
    x_plot = np.arange(X.shape[0])
    if X.shape[-1] == 1:
        plt.plot(x_plot, X, label='X')
    plt.plot(x_plot, y, label='y')
    plt.legend()
    plt.show()


def _standardize_or_to_array(variable, to_standardize, *dfs):
    if variable in to_standardize:
        return standardize(*dfs, return_scaler=False)
    return dfs_to_np_arrays(dfs)


def main():
    torch.set_default_dtype(torch.float32)
    print("loading data...")
    X_train, X_test, y_train, y_test, X, y = get_clean_data(N_POINTS_PER_GROUP, TO_STANDARDIZE, do_plot_data=PLOT_DATA)
    print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    print("running method...")
    X_uq = np.row_stack((X_train, X_test))

    y_pred, y_quantiles, y_std = run_mean_var_nn(
        X_train,
        y_train,
        X_uq,
        QUANTILES,
        n_iter=N_ITER,
        lr=LR,
        lr_patience=LR_PATIENCE,
        regularization=REGULARIZATION,
        warmup_period=WARMUP_PERIOD,
        frozen_var_value=FROZEN_VAR_VALUE,
        use_scheduler=USE_SCHEDULER,
    )

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
