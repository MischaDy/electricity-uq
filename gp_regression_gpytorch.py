import gpytorch
import torch

from helpers import tensor_to_numpy, standardize, numpy_to_tensor, df_to_tensor, get_data
from io_helper import IO_Helper

N_EPOCHS = 1000
LR = 1e-1  # 1e-3

N_DATAPOINTS = 800
STANDARDIZE_X = True
STANDARDIZE_Y = True
PRECOND_SIZE = 10

USE_SCHEDULER = True

SKIP_TRAINING = False

SHOW_PROGRESS = True
PLOT_LOSSES = True
PLOT_DATA = False

SHOW_UQ_PLOT = False
SAVE_UQ_PLOT = True
SAVE_MODEL = True
MODEL_NAME = 'gpytorch_model'

VALIDATION_FRAC = 0.1

IO_HELPER = IO_Helper('.')


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
def prepare_data(val_frac=0):
    X_train, X_test, y_train, y_test, X, y = get_data(
        N_DATAPOINTS,
        output_cols=['load_to_pred'],
        return_full_data=True
    )

    # split into train and val
    if val_frac > 0:
        n_samples = X_train.shape[0]
        val_size = max(1, round(val_frac * n_samples))
        train_size = max(1, n_samples - val_size)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train, y_train = X_train[:train_size], y_train[:train_size]
        assert X_train.shape[0] > 0 and X_val.shape[0] > 0
    else:
        X_val, y_val = None, None

    if STANDARDIZE_X:
        X_scaler, (X_train, X_val, X_test, X) = standardize(X_train, X_val, X_test, X, return_scaler=True)
        X_train, X_val, X_test, X = map(numpy_to_tensor, (X_train, X_val, X_test, X))
    else:
        X_train, X_val, X_test, X = map(df_to_tensor, (X_train, X_val, X_test, X))

    if STANDARDIZE_Y:
        y_scaler, (y_train, y_val, y_test, y) = standardize(y_train, y_val, y_test, y, return_scaler=True)
        y_train, y_val, y_test, y = map(numpy_to_tensor, (y_train, y_val, y_test, y))
    else:
        y_train, y_val, y_test, y = map(df_to_tensor, (y_train, y_val, y_test, y))

    y_train, y_val, y_test, y = map(lambda y: y.squeeze(), (y_train, y_val, y_test, y))

    shapes = map(lambda arr: arr.shape if arr is not None else None,
                 (X_train, y_train, X_val, y_val, X_test, y_test))
    print("data shapes:", ' - '.join(map(str, shapes)))

    if PLOT_DATA:
        print('plotting data...')
        plot_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # make continguous and map to device
    print('making data contiguous and mapping to device...')
    X_train, y_train, X_val, y_val, X_test, y_test = map(lambda tensor: tensor.contiguous() if tensor is not None else None,
                                                         (X_train, y_train, X_val, y_val, X_test, y_test))
    if torch.cuda.is_available():
        output_device = torch.device('cuda:0')
    else:
        print('warning: cuda unavailable!')
        output_device = torch.device('cpu')
    X_train, y_train, X_val, y_val, X_test, y_test = map(lambda tensor: tensor.to(output_device) if tensor is not None else None,
                                                         (X_train, y_train, X_val, y_val, X_test, y_test))
    return X_train, y_train, X_val, y_val, X_test, y_test


@measure_runtime
def train_gpytorch(
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        lr_patience=30,
        lr_reduction_factor=0.5,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_reduction_factor)
    use_scheduler = False if X_val is None else USE_SCHEDULER

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # with gpytorch.settings.max_preconditioner_size(preconditioner_size):

    losses = []
    epochs = np.arange(N_EPOCHS) + 1
    if SHOW_PROGRESS:
        epochs = tqdm(epochs)
    for epoch in epochs:
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = -mll(y_pred, y_train).sum()
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

        if use_scheduler:
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                y_pred = model.likelihood(model(X_val))
                val_loss = -mll(y_pred, y_val).sum()
            scheduler.step(val_loss)

    if PLOT_LOSSES:
        plot_skip_losses = 0
        plot_losses(losses[plot_skip_losses:])

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
        plot_name='gpytorch',
        show_plot=True,
        save_plot=True,
        plots_path='plots',
):
    import matplotlib.pyplot as plt
    import numpy as np

    X_train, X_val, X_test, y_train, y_val, y_test, y_preds, y_std = map(tensor_to_numpy, (X_train, X_val, X_test, y_train, y_val, y_test, y_preds, y_std))
    X_train = np.row_stack((X_train, X_val))
    y_train = np.hstack((y_train, y_val))
    num_train_steps, num_test_steps = X_train.shape[0], X_test.shape[0]

    x_plot_train = np.arange(num_train_steps)
    x_plot_full = np.arange(num_train_steps + num_test_steps)
    x_plot_test = np.arange(num_train_steps, num_train_steps + num_test_steps)
    x_plot_uq = x_plot_full

    ci_low, ci_high = y_preds - y_std / 2, y_preds + y_std / 2

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
    label = '1 std'
    ax.fill_between(
        x_plot_uq.ravel(),
        ci_low,
        ci_high,
        color="green",
        alpha=0.2,
        label=label,
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
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(val_frac=VALIDATION_FRAC)

    skip_training = SKIP_TRAINING
    common_prefix = f'{MODEL_NAME}_{N_DATAPOINTS}_{N_EPOCHS}'
    model_name, model_likelihood_name = f'{common_prefix}.pth', f'{common_prefix}_likelihood.pth'
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
        model, likelihood = train_gpytorch(X_train, y_train, X_val, y_val)

    # noinspection PyUnboundLocalVariable
    model.eval()
    # noinspection PyUnboundLocalVariable
    likelihood.eval()

    if SAVE_MODEL:
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
        X_uq = torch.row_stack((X_train, X_val, X_test) if X_val is not None else (X_train, X_test))
        f_preds = model(X_uq)
    f_mean = f_preds.mean
    y_preds = f_mean
    f_std = f_preds.stddev
    y_std = f_std

    # std2 = f_preds.stddev.mul_(2)
    # mean = self.mean
    # return mean.sub(std2), mean.add(std2)
    plot_uq_result(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        y_preds,
        y_std,
        plot_name='gpytorch',
        show_plot=SHOW_UQ_PLOT,
        save_plot=SAVE_UQ_PLOT,
        plots_path='plots',
    )


# def get_data(_n_points_per_group=None, filepath="data.pkl", input_cols=None, output_cols=None,
#              return_full_data=False, ):
#     df = pd.read_pickle(filepath)
#     if output_cols is None:
#         output_cols = ['load_to_pred']
#     if input_cols is None:
#         input_cols = [col for col in df.columns
#                       if col not in output_cols and not col.startswith('ts')]
#     lim = 2 * _n_points_per_group if _n_points_per_group is not None else -1
#     X = df[input_cols].iloc[:lim]
#     y = df[output_cols].iloc[:lim]
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.5, shuffle=False
#     )
#     if return_full_data:
#         return X_train, X_test, y_train, y_test, X, y
#     return X_train, X_test, y_train, y_test


def plot_data(X_train, y_train, X_val, y_val, X_test, y_test):
    import matplotlib.pyplot as plt
    import numpy as np

    if X_val is not None:
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
    plt.show()


def plot_losses(losses):
    import matplotlib.pyplot as plt

    def has_neg(losses):
        return any(map(lambda x: x < 0, losses))

    fig, ax = plt.subplots()
    plt_func = ax.plot if has_neg(losses) else ax.semilogy
    plt_func(losses, label="loss")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
