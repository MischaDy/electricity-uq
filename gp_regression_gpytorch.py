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

N_DATAPOINTS = 800
STANDARDIZE_X = True
STANDARDIZE_Y = True
PRECOND_SIZE = 0

SHOW_PROGRESS = True
PLOT_LOSSES = True
PLOT_DATA = False


def main():
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

    print("data shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    if PLOT_DATA:
        print('plotting data...')
        my_plot(X_train, X_test, y_train, y_test)

    # make continguous and map to device
    X_train, y_train = X_train.contiguous(), y_train.contiguous()
    X_test, y_test = X_test.contiguous(), y_test.contiguous()

    output_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    X_train, y_train = X_train.to(output_device), y_train.to(output_device)
    X_test, y_test = X_test.to(output_device), y_test.to(output_device)

    n_devices = torch.cuda.device_count()
    print('Planning to run on {} GPUs.'.format(n_devices))

    print('training...')
    model, likelihood = train(
        X_train,
        y_train,
        n_devices=n_devices,
        output_device=output_device,
        preconditioner_size=PRECOND_SIZE,
        n_epochs=N_EPOCHS,
        show_progress=SHOW_PROGRESS,
        do_plot_losses=PLOT_LOSSES,
    )

    # Get into evaluation (predictive posterior) mode
    print('evaluating...')
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions on a small number of test points to get the test time caches computed
        _ = model(X_test[:2, :])
        del _  # We don't care about these predictions, we really just want the caches.

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        t1 = default_timer()
        latent_pred = model(X_test)
        t2 = default_timer()
        print(f'cache stuff time: {t2 - t1}')

    print('computing loss...')
    test_rmse = torch.sqrt(torch.mean(torch.pow(latent_pred.mean - y_test, 2)))
    print(f"Test RMSE: {test_rmse.item()}")


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


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood, n_devices, output_device):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module,
            device_ids=range(n_devices),
            output_device=output_device,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(
        X_train,
        y_train,
        n_devices,
        output_device,
        preconditioner_size=15,
        n_epochs=100,
        show_progress=True,
        do_plot_losses=True,
):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
    model = ExactGPModel(X_train, y_train, likelihood, n_devices).to(output_device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.settings.max_preconditioner_size(preconditioner_size):
        def closure():
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            return loss

        loss = closure()
        loss.backward()

        losses = []
        epochs = np.arange(n_epochs) + 1
        if show_progress:
            epochs = tqdm(epochs)
        for epoch in epochs:
            loss = optimizer.step(closure=closure)
            losses.append(loss)

            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                epoch, n_epochs, loss,
                model.covar_module.module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))

        if do_plot_losses:
            plot_skip_losses = 0
            plot_losses(losses[plot_skip_losses:])

    print(f"Finished training on {X_train.size(0)} data points using {n_devices} GPUs.")
    model.eval()
    return model, likelihood


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
