import logging

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

from more_itertools import collapse
from tqdm import tqdm

from helpers import misc_helpers

torch.set_default_device(misc_helpers.get_device())
torch.set_default_dtype(torch.float32)


class MeanVarNN(torch.nn.Module):
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
        ])
        self.first_layer_stack = torch.nn.Sequential(*layers)
        self.last_layer_mean = torch.nn.Linear(hidden_layer_size, 1)
        self.last_layer_var = torch.nn.Linear(hidden_layer_size, 1)
        self._frozen_var = None

    def forward(self, x: torch.Tensor):
        x = self.first_layer_stack(x)
        mean = self.last_layer_mean(x)
        var = torch.exp(self.last_layer_var(x))
        if self._frozen_var is not None:
            var = torch.full(var.shape, self._frozen_var)
        return mean, var

    def freeze_variance(self, value: float):
        assert value > 0
        self._frozen_var = value

    def unfreeze_variance(self):
        self._frozen_var = None


def train_mean_var_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: MeanVarNN = None,
    n_iter=200,
    batch_size=20,
    num_hidden_layers=2,
    hidden_layer_size=50,
    activation=torch.nn.LeakyReLU,
    random_seed=42,
    lr=0.1,
    use_scheduler=True,
    lr_patience=30,
    lr_reduction_factor=0.5,
    weight_decay=0.0,
    do_train_var=True,
    frozen_var_value=0.5,
    loss_skip=10,
    show_progress_bar=True,
    show_losses_plot=True,
    save_losses_plot=True,
    io_helper=None,
):
    torch.manual_seed(random_seed)

    y_val_np = y_val.copy()  # for eval

    X_train, y_train, X_val, y_val = misc_helpers.np_arrays_to_tensors(X_train, y_train, X_val, y_val)
    X_train, y_train, X_val, y_val = misc_helpers.objects_to_cuda(X_train, y_train, X_val, y_val)
    X_train, y_train, X_val, y_val = misc_helpers.make_tensors_contiguous(X_train, y_train, X_val, y_val)

    dim_in, dim_out = X_train.shape[-1], y_train.shape[-1]

    if model is None:
        model = MeanVarNN(
            dim_in,
            num_hidden_layers=num_hidden_layers,
            hidden_layer_size=hidden_layer_size,
            activation=activation,
        )
    model = misc_helpers.object_to_cuda(model)

    # noinspection PyTypeChecker
    train_loader = misc_helpers.get_train_loader(X_train, y_train, batch_size)

    if do_train_var:
        model.unfreeze_variance()
        criterion = torch.nn.GaussianNLLLoss()
        criterion = misc_helpers.object_to_cuda(criterion)
    else:
        model.freeze_variance(frozen_var_value)
        _mse_loss = torch.nn.MSELoss()
        criterion = lambda input_, target, var: _mse_loss(input_, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_reduction_factor)

    train_losses, val_losses = [], []
    epochs = range(1, n_iter+1)
    if show_progress_bar:
        epochs = tqdm(epochs)
    for epoch in epochs:
        if not show_progress_bar:
            logging.info(f'epoch {epoch}/{n_iter}')
        model.train()
        for X_train, y_train in train_loader:
            optimizer.zero_grad()
            y_pred_mean, y_pred_var = model(X_train)
            loss = criterion(y_pred_mean, y_train, y_pred_var)
            loss.backward()
            optimizer.step()
        if not any([use_scheduler, show_losses_plot, save_losses_plot]):
            continue

        model.eval()
        with torch.no_grad():
            y_pred_mean_val, y_pred_var_val = model(X_val)
            val_loss = criterion(y_pred_mean_val, y_pred_var_val, y_val_np)
            if show_losses_plot or save_losses_plot:
                val_loss = misc_helpers.tensor_to_np_array(val_loss)
                val_losses.append(val_loss)

                y_pred_mean_train, y_pred_var_train = model(X_train)
                train_loss = criterion(y_pred_mean_train, y_train, y_pred_var_train)
                train_loss = misc_helpers.tensor_to_np_array(train_loss)
                train_losses.append(train_loss)
        if use_scheduler:
            scheduler.step(val_loss)
    misc_helpers.plot_nn_losses(
        train_losses,
        val_losses,
        loss_skip=loss_skip,
        show_plot=show_losses_plot,
        save_plot=save_losses_plot,
        io_helper=io_helper,
        filename='train_mean_var_nn',
    )
    model.eval()
    return model


def predict_with_mvnn(model, X_pred, quantiles):
    X_pred = misc_helpers.preprocess_array_to_tensor(X_pred)
    with torch.no_grad():
        y_pred, y_var = model(X_pred)
    y_pred, y_var = misc_helpers.tensors_to_np_arrays(y_pred, y_var)
    y_std = np.sqrt(y_var)
    y_quantiles = misc_helpers.quantiles_gaussian(quantiles, y_pred, y_std)
    return y_pred, y_quantiles, y_std
