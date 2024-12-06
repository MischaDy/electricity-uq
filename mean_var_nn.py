import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from matplotlib import pyplot as plt

from more_itertools import collapse
from tqdm import tqdm
from uncertainty_toolbox import nll_gaussian

from helpers import numpy_to_tensor, tensor_to_numpy, get_train_loader


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
        x = self.first_layer_stack(x)
        x = self.output_activation(x)
        return x

    @staticmethod
    def output_activation(x) -> tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: tensor of shape (n_samples, 2), where the dimension 1 are means and dimension 2 are variances
        :return:
        """
        mean, var = x[:, 0], x[:, 1]
        var = F.relu(var)
        return mean, var  # todo: possible?


def train_mean_var_nn(
    X,
    y,
    n_iter=10,
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
    iterable = tqdm(range(n_iter)) if verbose else range(n_iter)
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
    tensors = y_pred_mean, y_test, np.sqrt(y_pred_var)
    arrs = map(lambda t: t.numpy(force=True).squeeze(), tensors)
    return nll_gaussian(*arrs)


def predict(model, X, as_np=True):
    X = numpy_to_tensor(X)
    with torch.no_grad():
        tensor_pair = model(X)
    # tensor_pair = map(lambda t: t.reshape(-1, 1) if is_y_2d_ else t.squeeze(), tensor_pair)
    if as_np:
        return tuple(map(lambda t: tensor_to_numpy(t).squeeze(), tensor_pair))
    return tensor_pair


def native_mean_var_nn(
        X_train,
        y_train,
        n_iter=100,
        batch_size=20,
        random_state=711,
        verbose=True,
        val_frac=0.1,
        lr=0.1,
        lr_patience=5,
        lr_reduction_factor=0.5,
):
    X, y = ..., ...
    mean_var_nn = train_mean_var_nn(
        X,
        y,
        n_iter=n_iter,
        batch_size=batch_size,
        random_state=random_state,
        val_frac=val_frac,
        lr=lr,
        lr_patience=lr_patience,
        lr_reduction_factor=lr_reduction_factor,
        verbose=verbose,
    )

    if verbose:
        plot_post_training_perf(mean_var_nn, X_train, y_train)
    return mean_var_nn


def plot_post_training_perf(base_model, X_train, y_train):
    y_preds = base_model.predict(X_train)

    num_train_steps = X_train.shape[0]
    x_plot_train = np.arange(num_train_steps)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    ax.plot(x_plot_train, y_train, label='y_train', linestyle="dashed", color="black")
    ax.plot(
        x_plot_train,
        y_preds,
        label=f"base model prediction",
        color="green",
    )
    ax.legend()
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    plt.show()


if __name__ == '__main__':
    ...
