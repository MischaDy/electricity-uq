import gpytorch
import torch

from helpers import standardize, get_data, train_val_split, make_tensors_contiguous, \
    tensors_to_device, tensors_to_np_arrays, dfs_to_tensors, np_arrays_to_tensors, make_ys_1d
from io_helper import IO_Helper

N_EPOCHS = 100
LR = 1e-1  # 1e-3

N_DATAPOINTS = 100
STANDARDIZE_X = True
STANDARDIZE_Y = True
PRECOND_SIZE = 10

USE_SCHEDULER = True

SKIP_TRAINING = True

SHOW_PROGRESS = False
SHOW_PLOTS = False
PLOT_LOSSES = True
PLOT_DATA = False
SAVE_UQ_PLOT = True
SAVE_TRAINED = True
MODEL_NAME = 'gpytorch_model'

VALIDATION_FRAC = 0.1

IO_HELPER = IO_Helper('gpytorch_storage')


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel())

    def forward(self, X):
        X_mean = self.mean_module(X)
        X_covar = self.covar_module(X)
        return gpytorch.distributions.MultivariateNormal(X_mean, X_covar)


def measure_runtime(func):
    from timeit import default_timer

    def wrapper(*args, **kwargs):
        t1 = default_timer()
        result = func(*args, **kwargs)
        t2 = default_timer()
        print(f'done. [took {t2 - t1}s]')
        return result
    return wrapper


@measure_runtime
def preprocess_data(X_train, X_test, y_train, y_test, val_frac, standardize_x, standardize_y):
    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_frac=val_frac)

    if standardize_x:
        X_scaler, (X_train, X_val, X_test) = standardize(X_train, X_val, X_test, return_scaler=True)
        X_train, X_val, X_test = np_arrays_to_tensors(X_train, X_val, X_test)
    else:
        X_train, X_val, X_test = dfs_to_tensors(X_train, X_val, X_test)

    if standardize_y:
        y_scaler, (y_train, y_val, y_test) = standardize(y_train, y_val, y_test, return_scaler=True)
        y_train, y_val, y_test = np_arrays_to_tensors(y_train, y_val, y_test)
    else:
        y_train, y_val, y_test = dfs_to_tensors(y_train, y_val, y_test)

    y_train, y_val, y_test = make_ys_1d(y_train, y_val, y_test)

    shapes = map(lambda arr: arr.shape, (X_train, y_train, X_val, y_val, X_test, y_test))
    print("data shapes:", ' - '.join(map(str, shapes)))

    if PLOT_DATA:
        print('plotting data...')
        plot_data(X_train, y_train, X_val, y_val, X_test, y_test)

    print('making data contiguous and mapping to device...')
    X_train, y_train, X_val, y_val, X_test, y_test = make_tensors_contiguous(X_train, y_train, X_val, y_val, X_test, y_test)
    X_train, y_train, X_val, y_val, X_test, y_test = tensors_to_device(X_train, y_train, X_val, y_val, X_test, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


@measure_runtime
def train_gpytorch(
        X_train,
        y_train,
        X_val,
        y_val,
        n_epochs,
        use_scheduler=True,
        lr=1e-2,
        lr_patience=30,
        lr_reduction_factor=0.5,
        show_progress=True,
        show_plots=True,
        do_plot_losses=True,
):
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import numpy as np
    from tqdm import tqdm

    n_devices = torch.cuda.device_count()
    print('Planning to run on {} GPUs.'.format(n_devices))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)
    if torch.cuda.is_available():
        likelihood = likelihood.cuda()
        model = model.cuda()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_reduction_factor)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # with gpytorch.settings.max_preconditioner_size(preconditioner_size):

    losses = []
    epochs = np.arange(n_epochs) + 1
    if show_progress:
        epochs = tqdm(epochs)
    for epoch in epochs:
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

    print(f"Finished training on {X_train.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood


@measure_runtime
def evaluate(model, likelihood, X_test, y_test):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_pred = model.likelihood(model(X_test))

    print('computing loss...')
    rmse = (y_pred.mean - y_test).square().mean().sqrt().item()
    print(f"RMSE: {rmse:.3f}")
    return y_pred


def plot_uq_result(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        y_preds,
        y_std,
        n_stds=2,
        save_plot=True,
        show_plot=True,
        plot_name='gpytorch',
        plots_path='plots',
):
    import matplotlib.pyplot as plt
    import numpy as np

    X_train, X_val, X_test, y_train, y_val, y_test, y_preds, y_std = tensors_to_np_arrays(X_train, X_val, X_test, y_train, y_val, y_test, y_preds, y_std)
    X_train = np.row_stack((X_train, X_val))
    y_train = np.hstack((y_train, y_val))
    num_train_steps, num_test_steps = X_train.shape[0], X_test.shape[0]

    x_plot_train = np.arange(num_train_steps)
    x_plot_full = np.arange(num_train_steps + num_test_steps)
    x_plot_test = np.arange(num_train_steps, num_train_steps + num_test_steps)
    x_plot_uq = x_plot_full

    ci_low, ci_high = y_preds - n_stds * y_std, y_preds + n_stds * y_std

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    ax.plot(x_plot_train, y_train, label='y_train', linestyle="dashed", color="black")
    ax.plot(x_plot_test, y_test, label='y_test', linestyle="dashed", color="blue")
    ax.plot(
        x_plot_uq,
        y_preds,
        label=f"mean/median prediction {plot_name}",  # todo: mean or median?
        color="green",
    )
    # noinspection PyUnboundLocalVariable
    ax.fill_between(
        x_plot_uq.ravel(),
        ci_low,
        ci_high,
        color="green",
        alpha=0.2,
        label=f'+/- {n_stds} std',
    )
    ax.legend()
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title(plot_name)
    if save_plot:
        filename = f"{plot_name}.png"
        import os
        filepath = os.path.join(plots_path, filename)
        os.makedirs(plots_path, exist_ok=True)
        plt.savefig(filepath)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    print('preparing data...')
    X_train, X_test, y_train, y_test = get_data(
        N_DATAPOINTS,
        output_cols=['load_to_pred'],
        return_full_data=False,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        X_train,
        X_test,
        y_train,
        y_test,
        val_frac=VALIDATION_FRAC,
        standardize_x=STANDARDIZE_X,
        standardize_y=STANDARDIZE_Y,
    )

    skip_training = SKIP_TRAINING
    common_prefix, common_postfix = f'{MODEL_NAME}', f'{N_DATAPOINTS}_{N_EPOCHS}'
    model_name, model_likelihood_name = f'{common_prefix}_{common_postfix}.pth', f'{common_prefix}_likelihood_{common_postfix}.pth'
    if skip_training:
        print('skipping training...')
        try:
            likelihood = IO_HELPER.load_gpytorch_model(gpytorch.likelihoods.GaussianLikelihood, model_likelihood_name)
            model = IO_HELPER.load_gpytorch_model(ExactGPModel, model_name,
                                                  X_train=X_train, y_train=y_train, likelihood=likelihood)
        except FileNotFoundError:
            print(f'error: cannot load models {model_name} and/or {model_likelihood_name}')
            skip_training = False

    if not skip_training:
        print('training...')
        model, likelihood = train_gpytorch(
            X_train,
            y_train,
            X_val,
            y_val,
            use_scheduler=USE_SCHEDULER,
            n_epochs=N_EPOCHS,
            show_progress=SHOW_PROGRESS,
            show_plots=SHOW_PLOTS,
            do_plot_losses=PLOT_LOSSES,
            lr=LR,
        )

    # noinspection PyUnboundLocalVariable
    model.eval()
    # noinspection PyUnboundLocalVariable
    likelihood.eval()

    if SAVE_TRAINED:
        if skip_training:
            print('skipped training, so not saving models.')
        else:
            print('saving...')
            IO_HELPER.save_gpytorch_model(model, model_name)
            IO_HELPER.save_gpytorch_model(likelihood, model_likelihood_name)

    print('evaluating...')
    # noinspection PyUnboundLocalVariable
    evaluate(model, likelihood, X_test, y_test)

    print('plotting...')
    with torch.no_grad():
        X_uq = torch.row_stack((X_train, X_val, X_test))
        f_preds = model(X_uq)
    y_preds = f_preds.mean
    y_std = f_preds.stddev

    plot_uq_result(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        y_preds,
        y_std,
        save_plot=SAVE_UQ_PLOT,
        show_plot=SHOW_PLOTS,
        plot_name='gpytorch',
        plots_path='plots',
    )


def plot_data(X_train, y_train, X_val, y_val, X_test, y_test):
    import matplotlib.pyplot as plt
    import numpy as np

    X_train = np.row_stack((X_train, X_val))
    y_train = np.hstack((y_train, y_val))

    num_train_steps = X_train.shape[0]
    num_test_steps = X_test.shape[0]

    x_plot_train = np.arange(num_train_steps)
    x_plot_test = np.arange(num_test_steps) + num_train_steps
    plt.figure(figsize=(14, 6))
    plt.plot(x_plot_train, y_train, label='y_train')
    plt.plot(x_plot_test, y_test, label='y_test')
    plt.legend()
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_losses(losses, show_plots=True):
    import matplotlib.pyplot as plt

    def has_neg(losses):
        return any(map(lambda x: x < 0, losses))

    fig, ax = plt.subplots()
    plt_func = ax.plot if has_neg(losses) else ax.semilogy
    plt_func(losses, label="loss")
    ax.legend()
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
