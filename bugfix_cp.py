import numpy as np
import torch
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from skorch import NeuralNetRegressor
from torch import nn

from helpers import standardize, get_data, is_ascending
from temp_nn_file import My_NN

TO_STANDARDIZE = "xy"
QUANTILES = [0.05, 0.25, 0.75, 0.95]


def estimate_pred_interals_no_pfit_enbpi(model, cv_mapie_ts, alpha, X_test, X_train=None, y_train=None):
    alpha = list(alpha)
    mapie_enbpi = MapieTimeSeriesRegressor(model, method="enbpi", cv=cv_mapie_ts, agg_function="mean", n_jobs=-1)
    print("training...")
    mapie_enbpi = mapie_enbpi.fit(X_train, y_train)
    print("predicting...")
    y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit = mapie_enbpi.predict(
        X_test, alpha=alpha, ensemble=True, optimize_beta=False, allow_infinite_bounds=True
    )
    return y_pred_enbpi_no_pfit, y_pis_enbpi_no_pfit


def posthoc_conformal_prediction(X_train, y_train, X_uq, quantiles, model, random_state=42):
    cv = BlockBootstrap(n_resamplings=10, n_blocks=10, overlapping=False, random_state=random_state)
    alphas = pis_from_quantiles(quantiles)
    y_pred, y_pis = estimate_pred_interals_no_pfit_enbpi(model, cv, alphas, X_uq, X_train, y_train)
    y_quantiles = quantiles_from_pis(y_pis)  # (n_samples, 2 * n_intervals)
    return y_pred, y_quantiles, None


def pis_from_quantiles(quantiles):
    mid = len(quantiles) // 2
    first, second = quantiles[:mid], quantiles[mid:]
    pi_limits = zip(first, reversed(second))
    pis = [high - low for low, high in pi_limits]
    return sorted(pis)


def quantiles_from_pis(pis, check_order=False):
    if check_order:
        assert np.all([is_ascending(pi[0, :], reversed(pi[1, :])) for pi in pis])
    y_quantiles = np.array([sorted(pi.flatten()) for pi in pis])
    return y_quantiles


def my_train_base_model_nn(X_train, y_train, n_epochs=10, batch_size=1, random_state=711, lr=0.1,):
    torch.manual_seed(random_state)
    X_train, y_train = map(lambda arr: _arr_to_tensor(arr), (X_train, y_train))
    dim_in, dim_out = X_train.shape[-1], y_train.shape[0]
    model = NeuralNetRegressor(
        module=My_NN, criterion=nn.MSELoss(), optimizer=torch.optim.Adam, module__dim_in=dim_in,
        module__dim_out=dim_out, lr=lr, max_epochs=n_epochs, batch_size=batch_size,
        predict_nonlinearity=None, verbose=0,
    )
    model.fit(X_train, y_train)
    return model


def _arr_to_tensor(arr) -> torch.Tensor:
    return torch.Tensor(arr).float()


def run():
    print("loading data")
    X_train, X_test, y_train, y_test, X, y = my_get_data(_n_points_per_group=40)
    print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    base_model = my_train_base_model_nn(X_train, y_train)
    X_uq = np.row_stack((X_train, X_test))
    y_pred, y_quantiles, y_std = posthoc_conformal_prediction(
        X_train, y_train, X_uq, QUANTILES, base_model
    )
    return y_pred, y_quantiles, y_std


def my_get_data(_n_points_per_group=40):
    X_train, X_test, y_train, y_test, X, y = get_data(_n_points_per_group, return_full_data=True)
    X_train, X_test, X = _standardize_or_to_array("x", X_train, X_test, X)
    y_train, y_test, y = _standardize_or_to_array("y", y_train, y_test, y)
    y_train, y_test, y = map(lambda var: var.reshape(-1, 1), (y_train, y_test, y))
    return X_train, X_test, y_train, y_test, X, y


def _standardize_or_to_array(variable, *dfs):
    if variable in TO_STANDARDIZE:
        return standardize(False, *dfs)
    return map(lambda df: df.to_numpy(dtype=float), dfs)


if __name__ == '__main__':
    res = run()
    print(res)
