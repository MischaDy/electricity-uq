import logging
from typing import TYPE_CHECKING
from tqdm import tqdm
import gpytorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from helpers import misc_helpers

if TYPE_CHECKING:
    import numpy as np


torch.set_default_device(misc_helpers.get_device())
torch.set_default_dtype(torch.float32)


# noinspection PyPep8Naming
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel())

    def forward(self, X):
        X_mean = self.mean_module(X)
        X_covar = self.covar_module(X)
        return gpytorch.distributions.MultivariateNormal(X_mean, X_covar)


@misc_helpers.measure_runtime
def train_gpytorch(
        X_train,
        y_train,
        X_val,
        y_val,
        n_iter,
        use_scheduler=True,
        lr=1e-2,
        lr_patience=30,
        lr_reduction_factor=0.5,
        show_progress=True,
        show_plots=True,
        show_losses_plot=True,
        save_losses_plot=True,
        io_helper=None,
):
    n_devices = torch.cuda.device_count()
    logging.info('Planning to run on {} GPUs.'.format(n_devices))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)
    model, likelihood = misc_helpers.objects_to_cuda(model, likelihood)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_reduction_factor)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # with gpytorch.settings.max_preconditioner_size(preconditioner_size):
    train_losses, val_losses = [], []
    epochs = range(1, n_iter+1)
    if show_progress:
        epochs = tqdm(epochs)
    for epoch in epochs:
        if not show_progress:
            logging.info(f'epoch {epoch}/{n_iter}')
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = -mll(y_pred, y_train).sum()
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if use_scheduler or show_losses_plot:
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.memory_efficient(True):
                y_pred = model.likelihood(model(X_val))
                val_loss = -mll(y_pred, y_val).sum()
                val_losses.append(val_loss.item())
            scheduler.step(val_loss)

    misc_helpers.plot_nn_losses(
        train_losses,
        val_losses,
        loss_skip=0,
        show_plot=show_plots and show_losses_plot,
        save_plot=save_losses_plot,
        io_helper=io_helper,
        filename='train_gpytorch',
    )

    logging.info(f"Finished training on {X_train.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood


@misc_helpers.measure_runtime
def evaluate(model, likelihood, X_test, y_test):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.memory_efficient(True):
        y_pred = model.likelihood(model(X_test))

    logging.info('computing loss...')
    rmse = (y_pred.mean - y_test).square().mean().sqrt().item()
    logging.info(f"RMSE: {rmse:.3f}")
    return y_pred


def prepare_data(
        X_train: 'np.ndarray',
        y_train: 'np.ndarray',
        X_val: 'np.ndarray',
        y_val: 'np.ndarray',
        X_pred: 'np.ndarray',
):
    logging.info('preparing data..')
    X_train, y_train, X_val, y_val, X_pred = misc_helpers.np_arrays_to_tensors(
        X_train, y_train, X_val, y_val, X_pred
    )
    y_train, y_val = misc_helpers.make_arrs_1d(y_train, y_val)

    logging.info('mapping data to device and making it contiguous...')
    X_train, y_train, X_val, y_val, X_pred = misc_helpers.objects_to_cuda(X_train, y_train, X_val, y_val, X_pred)
    X_train, y_train, X_val, y_val, X_pred = misc_helpers.make_tensors_contiguous(
        X_train, y_train, X_val, y_val, X_pred
    )
    return X_train, y_train, X_val, y_val, X_pred


def predict_with_gpytorch(model, likelihood, X_pred, quantiles):
    model.eval()
    likelihood.eval()
    # todo: via gpytorch.settings, use fast_pred_samples, fast_computations?
    with torch.no_grad(), gpytorch.settings.memory_efficient(True):
        f_pred = model(X_pred)
    y_pred = f_pred.mean
    y_std = f_pred.stddev
    y_pred, y_std = misc_helpers.tensors_to_np_arrays(y_pred, y_std)
    y_quantiles = misc_helpers.quantiles_gaussian(quantiles, y_pred, y_std)
    return y_pred, y_quantiles, y_std
