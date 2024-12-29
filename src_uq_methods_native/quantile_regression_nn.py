import logging

logging.info('importing')

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

from more_itertools import collapse
from tqdm import tqdm

from helpers import misc_helpers

logging.info('done')

torch.set_default_device(misc_helpers.get_device())


DATA_PATH = '../data/data_1600.pkl'
QUANTILES = [0.05, 0.25, 0.75, 0.95]

STANDARDIZE_DATA = True
PLOT_DATA = False

N_POINTS_PER_GROUP = 800

N_ITER = 100
LR = 1e-4
REGULARIZATION = 0  # 1e-2
USE_SCHEDULER = False
LR_PATIENCE = 30
DO_PLOT_LOSSES = True


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

    def forward(self, x: torch.Tensor, as_dict=False):
        result = self.layer_stack(x)
        if as_dict:
            result = {quantile: result[:, i]
                      for i, quantile in enumerate(self.quantiles)}
        return result


class MultiPinballLoss:
    def __init__(self, quantiles, reduction='mean'):
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
        loss = torch.zeros_like(y_pred_quantiles, dtype=torch.float)
        for i, pinball_loss in enumerate(self.pinball_losses):
            loss[i] = pinball_loss(y_pred_quantiles[:, i:i+1], y_true)  # i+1 to ensure correct shape
        loss = _reduce_loss(loss, self.reduction)
        return loss


class PinballLoss:
    """
    copied with minor changes from: https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/pinball_loss.py
    """
    def __init__(self, quantile, reduction='none'):
        """

        :param quantile:
        :param reduction: one of: 'mean', 'sum', 'none'
        """
        assert 0 <= quantile <= 1
        assert reduction in {'mean', 'sum', 'none'}
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


def preprocess_data(X_train, y_train, val_frac=0.1):
    logging.info('train/val split')
    X_train, y_train, X_val, y_val = misc_helpers.train_val_split(X_train, y_train, val_frac)
    assert X_train.shape[0] > 0 and X_val.shape[0] > 0

    logging.info('preprocess arrays')
    X_train, y_train, X_val, y_val = misc_helpers.preprocess_arrays(X_train, y_train, X_val, y_val)
    return X_train, y_train, X_val, y_val


def train_qr_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantiles: list,
    n_iter=200,
    batch_size=20,
    random_seed=42,
    val_frac=0.1,
    lr=0.1,
    lr_patience=30,
    lr_reduction_factor=0.5,
    weight_decay=0.0,
    show_progress=True,
    do_plot_losses=True,
    loss_skip=10,
    use_scheduler=True,
):
    """

    :param X_train:
    :param y_train:
    :param quantiles: will be sorted internally
    :param n_iter:
    :param batch_size:
    :param random_seed:
    :param val_frac:
    :param lr:
    :param lr_patience:
    :param lr_reduction_factor:
    :param weight_decay:
    :param show_progress:
    :param do_plot_losses:
    :param loss_skip:
    :param use_scheduler:
    :return:
    """
    logging.info('setup')
    torch.manual_seed(random_seed)

    X_train, y_train, X_val, y_val = preprocess_data(X_train, y_train, val_frac=val_frac)

    dim_in, dim_out = X_train.shape[-1], y_train.shape[-1]
    quantiles = sorted(quantiles)

    logging.info('setup model')
    model = QR_NN(
        dim_in,
        quantiles,
        num_hidden_layers=2,
        hidden_layer_size=50,
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
    iterable = np.arange(n_iter) + 1
    if show_progress:
        iterable = tqdm(iterable)
    logging.info('training')
    for _ in iterable:
        model.train()
        for X_train, y_train in train_loader:
            optimizer.zero_grad()
            y_pred_quantiles = model(X_train)
            loss = criterion(y_pred_quantiles, y_train)
            loss.backward()
            optimizer.step()

        if not use_scheduler and not do_plot_losses:
            continue

        model.eval()
        with torch.no_grad():
            y_pred_quantiles_train = model(X_train)
            train_loss = criterion(y_pred_quantiles_train, y_train)
            train_losses.append(train_loss.item())

            y_pred_quantiles_val = model(X_val)
            val_loss = criterion(y_pred_quantiles_val, y_val)
            val_losses.append(val_loss.item())
        if use_scheduler:
            scheduler.step(val_loss)
    logging.info('done training')
    if do_plot_losses:
        logging.info('plotting losses')
        for loss_type, losses in {'train_losses': train_losses, 'val_losses': val_losses}.items():
            logging.info(loss_type, train_losses[:5], min(losses), max(losses), any(np.isnan(losses)))
        misc_helpers.plot_nn_losses(train_losses, val_losses, show_plots=do_plot_losses, loss_skip=loss_skip)
    model.eval()
    return model


def predict_with_qr_nn(model, X_pred):
    X_pred = misc_helpers.preprocess_array(X_pred)
    with torch.no_grad():
        # noinspection PyUnboundLocalVariable,PyCallingNonCallable
        y_quantiles_dict = model(X_pred, as_dict=True)
    y_quantiles = np.array(list(y_quantiles_dict.values())).T
    y_pred = y_quantiles_dict[0.5]
    y_std = misc_helpers.stds_from_quantiles(y_quantiles)
    return y_pred, y_quantiles, y_std
