import logging

import numpy as np
import torch
from laplace import Laplace
from tqdm import tqdm

from helpers import misc_helpers

torch.set_default_dtype(torch.float32)


def train_laplace_approximation(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        base_model_nn,
        n_iter,
        batch_size=20,
        random_seed=42,
        verbose=True,
        show_progress_bar=True,
):
    # todo: offer option to alternatively optimize parameters and hyperparameters of the prior jointly (cf. example
    #  script)?
    torch.manual_seed(random_seed)
    torch.set_default_device(misc_helpers.get_device())

    X_train, y_train = misc_helpers.add_val_to_train(X_train, X_val, y_train, y_val)
    X_train, y_train = misc_helpers.preprocess_arrays_to_tensors(X_train, y_train)
    train_loader = misc_helpers.get_train_loader(X_train, y_train, batch_size)
    model = la_instantiator(base_model_nn)
    model.fit(train_loader)

    log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    epochs = tqdm(range(n_iter)) if show_progress_bar else range(n_iter)
    for epoch in epochs:
        if verbose and not show_progress_bar:
            logging.info(f'epoch {epoch}/{n_iter}')
        hyper_optimizer.zero_grad()
        neg_marglik = -model.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()
    return model


def predict_with_laplace_approximation(model, X_pred: np.ndarray, quantiles: list):
    logging.info('predicting...')
    X_pred = misc_helpers.preprocess_array_to_tensor(X_pred)
    f_mu, f_var = model(X_pred)
    f_mu = misc_helpers.tensor_to_np_array(f_mu.squeeze())
    f_sigma = misc_helpers.tensor_to_np_array(f_var.squeeze().sqrt())
    pred_std = np.sqrt(f_sigma ** 2 + model.sigma_noise.item() ** 2)

    y_pred, y_std = f_mu, pred_std
    y_quantiles = misc_helpers.quantiles_gaussian(quantiles, y_pred, y_std)
    return y_pred, y_quantiles, y_std


def la_instantiator(base_model: torch.nn.Module):
    return Laplace(base_model, "regression")
