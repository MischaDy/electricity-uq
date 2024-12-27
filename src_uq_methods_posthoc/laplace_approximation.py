import numpy as np
import torch
from laplace import Laplace
from torch import nn
from tqdm import tqdm

from helpers import misc_helpers


def train_laplace_approximation(
        X_train,
        y_train,
        base_model_nn,
        n_iter,
        batch_size=20,
        random_seed=42,
        verbose=True,
):
    # todo: offer option to alternatively optimize parameters and hyperparameters of the prior jointly (cf. example
    #  script)?

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(random_seed)
    torch.set_default_device(misc_helpers.get_device())

    X_train, y_train = misc_helpers.preprocess_arrays(X_train, y_train)
    train_loader = misc_helpers.get_train_loader(X_train, y_train, batch_size)
    la = la_instantiator(base_model_nn)
    la.fit(train_loader)

    log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    iterable = tqdm(range(n_iter)) if verbose else range(n_iter)
    for _ in iterable:
        hyper_optimizer.zero_grad()
        neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()


def predict_with_laplace_approximation(model, X_pred, quantiles):
    # noinspection PyArgumentList,PyUnboundLocalVariable
    f_mu, f_var = model(X_pred)
    f_mu = misc_helpers.tensor_to_np_array(f_mu.squeeze())
    f_sigma = misc_helpers.tensor_to_np_array(f_var.squeeze().sqrt())
    pred_std = np.sqrt(f_sigma ** 2 + model.sigma_noise.item() ** 2)

    y_pred, y_std = f_mu, pred_std
    y_quantiles = misc_helpers.quantiles_gaussian(quantiles, y_pred, y_std)
    return y_pred, y_quantiles, y_std


def la_instantiator(base_model: nn.Module):
    return Laplace(base_model, "regression")
