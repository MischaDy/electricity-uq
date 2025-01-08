import logging
from typing import Literal

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

from more_itertools import collapse
from tqdm import tqdm

from helpers import misc_helpers
from helpers.io_helper import IO_Helper

torch.set_default_device(misc_helpers.get_device())
torch.set_default_dtype(torch.float32)

USE_NEW_LOSS = False
IO_HELPER = IO_Helper('comparison_storage')


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
    # NN: 28s (1.8it/s) for n=50
    # QR: 30s (1.7it/s) for n=50, qs=7
    # QR: 38s (1.4it/s) for n=50, qs=99
    def __init__(self, quantiles, reduction: Literal['mean', 'sum', 'none'] = 'mean'):
        if list(quantiles) != sorted(quantiles):
            raise ValueError(f'Quantiles must be sorted: {quantiles}')
        assert all(map(lambda q: 0 < q < 1, quantiles))

        self.quantiles = quantiles
        self._quantiles_torch = torch.Tensor(quantiles).requires_grad_(False).reshape(1, -1)
        self._1_m_quantiles_torch = 1 - self._quantiles_torch
        self.reduction = reduction

    def __call__(self, y_pred_quantiles: torch.Tensor, y_true: torch.Tensor):
        assert y_pred_quantiles.shape[1] == len(self.quantiles)
        assert y_pred_quantiles.shape[0] == y_true.shape[0]

        # try to compute as in https://scikit-learn.org/stable/modules/model_evaluation.html#pinball-loss
        zeros = torch.zeros_like(y_pred_quantiles, requires_grad=False)
        zeros = misc_helpers.object_to_cuda(zeros)

        y_minus_q = y_true - y_pred_quantiles
        err_alpha = torch.max(zeros, y_minus_q)
        q_minus_y = -y_minus_q
        err_1_m_alpha = torch.max(zeros, q_minus_y)

        loss = self._quantiles_torch * err_alpha + self._1_m_quantiles_torch * err_1_m_alpha
        loss = _reduce_loss(loss, self.reduction)
        return loss

    def to(self, device):
        self._quantiles_torch = self._quantiles_torch.to(device)
        self._1_m_quantiles_torch = self._1_m_quantiles_torch.to(device)
        return self


class MultiPinballLossOld:
    def __init__(self, quantiles, reduction: Literal['mean', 'sum', 'none'] = 'mean'):
        if list(quantiles) != sorted(quantiles):
            raise ValueError(f'Quantiles must be sorted: {quantiles}')
        self.quantiles = quantiles
        self.reduction = reduction
        self.pinball_losses = [PinballLoss(quantile, reduction)
                               for quantile in self.quantiles]

    def __call__(self, y_pred_quantiles: torch.Tensor, y_true: torch.Tensor):
        assert y_pred_quantiles.shape[1] == len(self.pinball_losses)
        loss = torch.zeros(len(self.pinball_losses), dtype=torch.float)
        for i, pinball_loss in enumerate(self.pinball_losses):
            loss[i] = pinball_loss(y_pred_quantiles[:, i:i + 1], y_true)  # i+1 to ensure correct shape
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
        activation=torch.nn.LeakyReLU,
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

    logging.info('setup meta-models')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_reduction_factor)
    if USE_NEW_LOSS:
        criterion = MultiPinballLoss(quantiles, reduction='mean')
    else:
        criterion = MultiPinballLossOld(quantiles, reduction='mean')

    logging.info('map to cuda')
    model, criterion = misc_helpers.objects_to_cuda(model, criterion)

    logging.info('setup training')
    # noinspection PyTypeChecker
    train_loader = misc_helpers.get_train_loader(X_train, y_train, batch_size)
    train_losses, val_losses = [], []
    epochs = range(1, n_iter + 1)
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


def test_qr():
    import settings

    logging.basicConfig(level=logging.INFO)
    logging.info('data setup...')

    SHOW_PLOT = True
    SAVE_PLOT = True
    PLOT_DATA = False
    USE_REAL_DATA = True

    n_iter = 50
    n_samples = 1600
    train_frac = 0.4
    val_frac = 0.1
    test_frac = 0.5

    quantiles = settings.QUANTILES  # [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]

    n_train_samples = round(train_frac * n_samples)
    n_val_samples = round(val_frac * n_samples)
    n_test_samples = round(test_frac * n_samples)

    if USE_REAL_DATA:
        X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers.get_data(
            '../data/data_1600.pkl',
            n_points_per_group=n_samples,
        )
    else:
        dim = 10
        X = np.arange(n_samples * dim).reshape(n_samples, dim)
        y = np.sin(X.sum(axis=1) / n_samples / 3).reshape(-1, 1)

    if PLOT_DATA:
        logging.info('plotting data')
        from matplotlib import pyplot as plt
        plt.plot(y)
        plt.show(block=True)

    X_train = X[:n_train_samples]
    y_train = y[:n_train_samples]
    X_val = X[n_train_samples:n_train_samples + n_val_samples]
    y_val = y[n_train_samples:n_train_samples + n_val_samples]
    X_test = X[-n_test_samples:]
    y_test = y[-n_test_samples:]

    X_pred = X
    y_true = y

    kwargs = {
        "n_iter": n_iter,
        "num_hidden_layers": 2,
        "hidden_layer_size": 50,
        'random_seed': 42,
        'lr': 1e-4,
        'use_scheduler': False,
        'lr_patience': 30,
        "weight_decay": 1e-3,
        'show_progress_bar': True,
        'show_losses_plot': False,
        'save_losses_plot': False,
        'io_helper': IO_HELPER,
        'activation': torch.nn.LeakyReLU,
    }

    model = train_qr_nn(
        X_train,
        y_train,
        X_val,
        y_val,
        quantiles,
        **kwargs
    )
    y_pred, y_quantiles, y_std = predict_with_qr_nn(model, X_pred)
    ci_low, ci_high = (
        y_quantiles[:, 0],
        y_quantiles[:, -1],
    )
    n_quantiles = y_quantiles.shape[1]
    plot_uq_worker(y_true, y_pred, ci_low, ci_high, 'full', 'qr', n_quantiles, show_plot=SHOW_PLOT, save_plot=SAVE_PLOT)


def plot_uq_worker(y_true_plot, y_pred_plot, ci_low_plot, ci_high_plot, label_part,
                   method, n_quantiles=None, show_plot=True, save_plot=True):
    from matplotlib import pyplot as plt
    base_title = method
    base_filename = method
    label = f'outermost 2/{n_quantiles} quantiles' if n_quantiles is not None else 'outermost 2 quantiles'
    x_plot = np.arange(y_true_plot.shape[0])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    ax.plot(x_plot, y_true_plot, label=f'{label_part} data', color="black", linestyle='dashed')
    ax.plot(x_plot, y_pred_plot, label="point prediction", color="green")
    ax.fill_between(
        x_plot,
        ci_low_plot,
        ci_high_plot,
        color="green",
        alpha=0.2,
        label=label,
    )
    ax.legend()
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title(f'{base_title} ({label_part})')
    if save_plot:
        IO_HELPER.save_plot(filename=f'{base_filename}_{label_part}')
    if show_plot:
        plt.show(block=True)
    plt.close(fig)


if __name__ == '__main__':
    test_qr()
