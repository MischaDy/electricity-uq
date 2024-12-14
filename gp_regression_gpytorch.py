from timeit import default_timer

import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
from tqdm import tqdm

N_EPOCHS = 100
LR = 1e-3

N_DATAPOINTS = 800
STANDARDIZE_X = True
STANDARDIZE_Y = True
PRECOND_SIZE = 0

SHOW_PROGRESS = True
PLOT_LOSSES = True
PLOT_DATA = False


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def measure_runtime(func):
    def wrapper(*args, **kwargs):
        t1 = default_timer()
        result = func(*args, **kwargs)
        t2 = default_timer()
        print(f'done. [took {t2 - t1}s]')
        return result
    return wrapper


@measure_runtime
def prepare_data():
    X_train, X_test, y_train, y_test, X, y = get_data(
        N_DATAPOINTS,
        output_cols=['load_to_pred'],
        return_full_data=True
    )

    if STANDARDIZE_X:
        X_scaler, (X_train, X_test, X) = standardize(X_train, X_test, X)
        X_train, X_test, X = map(arr_to_tensor, (X_train, X_test, X))
    else:
        X_train, X_test, X = map(df_to_tensor, (X_train, X_test, X))

    if STANDARDIZE_Y:
        y_scaler, (y_train, y_test, y) = standardize(y_train, y_test, y)
        y_train, y_test, y = map(arr_to_tensor, (y_train, y_test, y))
    else:
        y_train, y_test, y = map(df_to_tensor, (y_train, y_test, y))

    y_train, y_test = map(lambda y: y.squeeze(), (y_train, y_test))
    print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    if PLOT_DATA:
        print('plotting data...')
        my_plot(X_train, X_test, y_train, y_test)

    # make continguous and map to device
    print('making data contiguous and mapping to device...')
    X_train, y_train = X_train.contiguous(), y_train.contiguous()
    X_test, y_test = X_test.contiguous(), y_test.contiguous()

    if torch.cuda.is_available():
        output_device = torch.device('cuda:0')
    else:
        print('warning: cuda unavailable!')
        output_device = torch.device('cpu')
    X_train, y_train = X_train.to(output_device), y_train.to(output_device)
    X_test, y_test = X_test.to(output_device), y_test.to(output_device)

    return X_train, y_train, X_test, y_test


@measure_runtime
def train(X_train, y_train):
    n_devices = torch.cuda.device_count()
    print('Planning to run on {} GPUs.'.format(n_devices))

    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = ExactGPModel(X_train, y_train, likelihood).cuda()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # with gpytorch.settings.max_preconditioner_size(preconditioner_size):

    losses = []
    epochs = np.arange(N_EPOCHS) + 1
    if SHOW_PROGRESS:
        epochs = tqdm(epochs)
    for epoch in epochs:
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train).sum()
        losses.append(loss.item())
        print_values = dict(
            loss=loss.item(),
            ls=model.covar_module.base_kernel.lengthscale.norm().item(),
            os=model.covar_module.outputscale.item(),
            noise=model.likelihood.noise.item(),
            mu=model.mean_module.constant.item(),
        )
        epochs.set_postfix(**print_values)
        loss.backward()
        optimizer.step()

    if PLOT_LOSSES:
        plot_skip_losses = 0
        plot_losses(losses[plot_skip_losses:])

    print(f"Finished training on {X_train.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood


@measure_runtime
def evaluate(model, likelihood, X_test, y_test):
    model.eval()
    # likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_pred = model.likelihood(model(X_test))

    print('computing loss...')
    rmse = (y_pred.mean - y_test).square().mean().sqrt().item()
    print(f"RMSE: {rmse:.3f}")


def main():
    print('preparing data...')
    X_train, y_train, X_test, y_test = prepare_data()

    print('training...')
    model, likelihood = train(X_train, y_train)

    print('evaluating...')
    evaluate(model, likelihood, X_test, y_test)
    return model, likelihood


def df_to_tensor(df: pd.DataFrame, dtype=float) -> torch.Tensor:
    return torch.Tensor(df.to_numpy(dtype=dtype))


def arr_to_tensor(arr):
    return torch.Tensor(arr).float()


def get_data(_n_points_per_group=None, filepath="data.pkl", input_cols=None, output_cols=None,
             return_full_data=False, ):
    df = pd.read_pickle(filepath)
    if output_cols is None:
        output_cols = ['load_to_pred']
    if input_cols is None:
        input_cols = [col for col in df.columns
                      if col not in output_cols and not col.startswith('ts')]
    lim = 2 * _n_points_per_group if _n_points_per_group is not None else -1
    X = df[input_cols].iloc[:lim]
    y = df[output_cols].iloc[:lim]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )
    if return_full_data:
        return X_train, X_test, y_train, y_test, X, y
    return X_train, X_test, y_train, y_test


def standardize(train_data, *arrays_to_standardize):
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler, map(scaler.transform, [train_data, *arrays_to_standardize])


def my_plot(X_train, X_test, y_train, y_test):
    num_train_steps = X_train.shape[0]
    num_test_steps = X_test.shape[0]

    x_plot_train = np.arange(num_train_steps)
    x_plot_test = np.arange(num_train_steps, num_train_steps + num_test_steps)
    plt.figure(figsize=(14, 6))
    plt.plot(x_plot_train, y_train, label='y_train')
    plt.plot(x_plot_test, y_test, label='y_test')
    plt.legend()
    plt.show()


def plot_losses(losses):
    def has_neg(losses):
        return any(map(lambda x: x < 0, losses))

    fig, ax = plt.subplots()
    plt_func = ax.plot if has_neg(losses) else ax.semilogy
    plt_func(losses, label="loss")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
