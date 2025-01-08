import logging
from typing import Literal

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

from more_itertools import collapse
from tqdm import tqdm

from helpers import misc_helpers

torch.set_default_device(misc_helpers.get_device())
torch.set_default_dtype(torch.float32)


class QR_NN(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        quantiles,
        num_hidden_layers=2,
        hidden_layer_size=50,
        activation=torch.nn.LeakyReLU,
    ):
        """

        :param dim_in:
        :param quantiles: the quantile levels to predict
        :param num_hidden_layers:
        :param hidden_layer_size:
        :param activation:
        """
        super().__init__()
        self.quantiles = quantiles
        dim_out = len(quantiles)
        layers = collapse([
            torch.nn.Linear(dim_in, hidden_layer_size),
            activation(),
            [[torch.nn.Linear(hidden_layer_size, hidden_layer_size),
              activation()]
             for _ in range(num_hidden_layers)],
            torch.nn.Linear(hidden_layer_size, dim_out),
        ])
        self.layer_stack = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, as_dict=False) -> torch.Tensor | dict[float, torch.Tensor]:
        result = self.layer_stack(x)
        if as_dict:
            result = {quantile: result[:, i]
                      for i, quantile in enumerate(self.quantiles)}
        return result


class MultiPinballLoss:
    def __init__(self, quantiles, reduction: Literal['mean', 'sum', 'none'] = 'mean'):
        if list(quantiles) != sorted(quantiles):
            raise ValueError(f'Quantiles must be sorted: {quantiles}')
        self.quantiles = quantiles
        self.reduction = reduction
        self.pinball_losses = [PinballLoss(quantile, reduction)
                               for quantile in self.quantiles]

    def __call__(self, y_pred_quantiles: torch.Tensor, y_true: torch.Tensor):
        # todo: optimize (with torch.vmap?)
        # todo: treat losses individually?
        assert y_pred_quantiles.shape[1] == len(self.pinball_losses)
        loss = torch.zeros(len(self.pinball_losses), dtype=torch.float)
        for i, pinball_loss in enumerate(self.pinball_losses):
            loss[i] = pinball_loss(y_pred_quantiles[:, i:i+1], y_true)  # i+1 to ensure correct shape
        loss = _reduce_loss(loss, self.reduction)
        return loss


class PinballLoss:
    """
    copied with minor changes from: https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/pinball_loss.py
    """
    def __init__(self, quantile: float, reduction: Literal['mean', 'sum', 'none'] = 'mean'):
        assert 0 <= quantile <= 1
        self.quantile = quantile
        self.reduction = reduction

    def __call__(self, y_pred_quantile, y_true):
        assert y_pred_quantile.shape == y_true.shape
        loss = torch.zeros_like(y_true, dtype=torch.float)
        error = y_pred_quantile - y_true
        smaller_index = error < 0
        bigger_index = 0 < error
        abs_error = abs(error)
        loss[smaller_index] = self.quantile * abs_error[smaller_index]
        loss[bigger_index] = (1 - self.quantile) * abs_error[bigger_index]
        loss = _reduce_loss(loss, self.reduction)
        return loss


def _reduce_loss(loss, reduction):
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()
    return loss


def compute_eval_losses(model, criterion, X_train, y_train, X_val, y_val):
    model.eval()
    with torch.no_grad():
        y_pred_quantiles_train = model(X_train)
        train_loss = criterion(y_pred_quantiles_train, y_train)
        y_pred_quantiles_val = model(X_val)
        val_loss = criterion(y_pred_quantiles_val, y_val)
    return train_loss, val_loss


def train_qr_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantiles: list,
    n_iter=200,
    batch_size=20,
    num_hidden_layers=2,
    hidden_layer_size=50,
    activation=None,
    random_seed=42,
    lr=0.1,
    use_scheduler=True,
    lr_patience=30,
    lr_reduction_factor=0.5,
    weight_decay=0.0,
    show_progress_bar=True,
    show_plots=True,
    show_losses_plot=True,
    save_losses_plot=True,
    io_helper=None,
    loss_skip=10,
):
    """

    :param num_hidden_layers:
    :param hidden_layer_size:
    :param activation:
    :param y_val:
    :param X_val:
    :param io_helper:
    :param show_losses_plot:
    :param show_plots:
    :param X_train:
    :param y_train:
    :param quantiles: will be sorted internally
    :param n_iter:
    :param batch_size:
    :param random_seed:
    :param lr:
    :param lr_patience:
    :param lr_reduction_factor:
    :param weight_decay:
    :param show_progress_bar:
    :param save_losses_plot:
    :param loss_skip:
    :param use_scheduler:
    :return:
    """
    logging.info('setup')
    torch.manual_seed(random_seed)

    X_train, y_train, X_val, y_val = misc_helpers.preprocess_arrays_to_tensors(X_train, y_train, X_val, y_val)

    dim_in, dim_out = X_train.shape[-1], y_train.shape[-1]
    quantiles = sorted(quantiles)

    logging.info('setup model')
    model = QR_NN(
        dim_in,
        quantiles,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_size=hidden_layer_size,
        activation=activation,
    )
    model = misc_helpers.object_to_cuda(model)

    logging.info('setup meta-models')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_reduction_factor)
    criterion = MultiPinballLoss(quantiles, reduction='mean')

    logging.info('setup training')
    # noinspection PyTypeChecker
    train_loader = misc_helpers.get_train_loader(X_train, y_train, batch_size)
    train_losses, val_losses = [], []
    epochs = range(1, n_iter+1)
    if show_progress_bar:
        epochs = tqdm(epochs)
    logging.info('training...')
    for epoch in epochs:
        if not show_progress_bar:
            logging.info(f'epoch {epoch}/{n_iter}')
        model.train()
        for X_train, y_train in train_loader:
            optimizer.zero_grad()
            y_pred_quantiles = model(X_train)
            loss = criterion(y_pred_quantiles, y_train)
            loss.backward()
            optimizer.step()
        if not use_scheduler and not save_losses_plot:
            continue

        train_loss, val_loss = compute_eval_losses(model, criterion, X_train, y_train, X_val, y_val)
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        if use_scheduler:
            scheduler.step(val_loss)
    logging.info('done training.')
    misc_helpers.plot_nn_losses(
        train_losses,
        val_losses,
        loss_skip=loss_skip,
        show_plot=show_plots and show_losses_plot,
        save_plot=save_losses_plot,
        io_helper=io_helper,
        filename='train_qr_nn',
    )
    model.eval()
    return model


def predict_with_qr_nn(model: QR_NN, X_pred: np.array):
    X_pred = misc_helpers.preprocess_array_to_tensor(X_pred)
    with torch.no_grad():
        y_quantiles_dict = model(X_pred, as_dict=True)
        y_quantiles_dict = {
            quantile: misc_helpers.tensor_to_np_array(tensor)
            for quantile, tensor in y_quantiles_dict.items()
        }
    y_quantiles = np.array(list(y_quantiles_dict.values())).T
    y_pred = y_quantiles_dict[0.5]
    y_std = misc_helpers.stds_from_quantiles(y_quantiles)
    return y_pred, y_quantiles, y_std
