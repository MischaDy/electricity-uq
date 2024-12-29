import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import gpytorch
import numpy as np
import torch

from helpers import misc_helpers

torch.set_default_device(misc_helpers.get_device())


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
        do_plot_losses=True,
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
    losses = []
    epochs = np.arange(n_iter) + 1
    if show_progress:
        epochs = tqdm(epochs)
    for _ in epochs:
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = -mll(y_pred, y_train).sum()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if use_scheduler:
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                y_pred = model.likelihood(model(X_val))
                val_loss = -mll(y_pred, y_val).sum()
            scheduler.step(val_loss)

    if do_plot_losses:
        plot_skip_losses = 0
        plot_losses(losses[plot_skip_losses:], show_plots=show_plots)

    logging.info(f"Finished training on {X_train.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood


@misc_helpers.measure_runtime
def evaluate(model, likelihood, X_test, y_test):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_pred = model.likelihood(model(X_test))

    logging.info('computing loss...')
    rmse = (y_pred.mean - y_test).square().mean().sqrt().item()
    logging.info(f"RMSE: {rmse:.3f}")
    return y_pred


def prepare_data(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_pred: np.ndarray,
        val_frac=0.1,
):
    logging.info('preparing data..')
    X_train, y_train, X_val, y_val = misc_helpers.train_val_split(X_train, y_train, val_frac=val_frac)
    X_train, y_train, X_val, y_val, X_pred = misc_helpers.np_arrays_to_tensors(
        X_train, y_train, X_val, y_val, X_pred
    )
    y_train, y_val = misc_helpers.make_ys_1d(y_train, y_val)

    logging.info('mapping data to device and making it contiguous...')
    X_train, y_train, X_val, y_val, X_pred = misc_helpers.objects_to_cuda(X_train, y_train, X_val, y_val, X_pred)
    X_train, y_train, X_val, y_val, X_pred = misc_helpers.make_tensors_contiguous(
        X_train, y_train, X_val, y_val, X_pred
    )
    return X_train, y_train, X_val, y_val, X_pred


def predict_with_gpytorch(model, likelihood, X_pred, quantiles):
    # noinspection PyUnboundLocalVariable
    model.eval()
    # noinspection PyUnboundLocalVariable
    likelihood.eval()
    with torch.no_grad():  # todo: use gpytorch.settings.fast_pred_var()?
        f_preds = model(X_pred)
    y_preds = f_preds.mean
    y_std = f_preds.stddev
    y_preds, y_std = misc_helpers.tensors_to_np_arrays(y_preds, y_std)
    y_quantiles = misc_helpers.quantiles_gaussian(quantiles, y_preds, y_std)
    return y_preds, y_quantiles, y_std


def plot_losses(losses, show_plots=True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt_func = ax.plot if _contains_neg(losses) else ax.semilogy
    plt_func(losses, label="loss")
    ax.legend()
    if show_plots:
        plt.show(block=True)
    plt.close(fig)


def _contains_neg(losses):
    return any(map(lambda x: x < 0, losses))
