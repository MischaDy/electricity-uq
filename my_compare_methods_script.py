import torch
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import randint, norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from mapie.subsample import BlockBootstrap
from laplace import Laplace

import time
from tqdm import tqdm
from typing import Any

from compare_methods import UQ_Comparer
from conformal_prediction import estimate_pred_interals_no_pfit_enbpi
from mean_var_nn import run_mean_var_nn
from quantile_regression import estimate_quantiles as estimate_quantiles_qr
from nn_estimator import NN_Estimator

from helpers import get_data, standardize, numpy_to_tensor, df_to_numpy, get_train_loader, tensor_to_numpy, tensor_to_device
from metrics import rmse, smape_scaled, crps, nll_gaussian, mean_pinball_loss


METHOD_WHITELIST = [
#    "posthoc_conformal_prediction",
    "posthoc_laplace",
#    "native_quantile_regression",
#    "native_gp",
#    "native_mvnn",
]
QUANTILES = [
    0.05,
    0.25,
    0.75,
    0.95,
]  # todo: how to handle 0.5? ==> just use mean if needed

DATA_FILEPATH = './data.pkl'

N_POINTS_PER_GROUP = 100  # 800
PLOT_DATA = False
PLOT_RESULTS = True
SHOW_PLOTS = False
SAVE_PLOTS = True

TEMP_TEST_ALL = False

PLOTS_PATH = "plots"

METHODS_KWARGS = {
    "native_mvnn": {
        "n_iter": 100,  # 300,
        "lr": 1e-4,
        "lr_patience": 30,
        "regularization": 0,  # 1e-2,
        "warmup_period": 50,
        "frozen_var_value": 0.1,
    },
    "native_quantile_regression": {
        "verbose": True,
    },
    "native_gp": {
        'n_restarts_optimizer': 5,
        "verbose": True,
    },
    "posthoc_conformal_prediction": {
        "n_estimators": 5,
        "verbose": 1,
        "skip_training": False,
    },
    "posthoc_laplace": {
        "n_iter": 100,  #300,
    },
    "base_model": {
        "n_iter": 100,  # 200,
        "skip_training": False,
        "save_trained": True,
        "verbose": 1,
    },
}

TO_STANDARDIZE = "xy"

torch.set_default_dtype(torch.float32)


# noinspection PyPep8Naming
class My_UQ_Comparer(UQ_Comparer):
    def __init__(
        self,
        storage_path="comparison_storage",
        to_standardize="X",
        methods_kwargs: dict[str, dict[str, Any]] = None,
        n_points_per_group=800,
        *args,
        **kwargs
    ):
        """

        :param methods_kwargs: dict of (method_name, method_kwargs_dict) pairs
        :param storage_path:
        :param to_standardize: iterable of variables to standardize. Can contain 'x' and/or 'y', or neither.
        :param args: passed to super.__init__
        :param kwargs: passed to super.__init__
        :param n_points_per_group: both training size and test size
        """
        super().__init__(*args, **kwargs)
        if methods_kwargs is not None:
            self.methods_kwargs.update(methods_kwargs)
        self.io_helper = IO_Helper(storage_path)
        self.to_standardize = to_standardize
        self.n_points_per_group = n_points_per_group

    # todo: remove param?
    def get_data(self):
        """

        :param _n_points_per_group:
        :return: X_train, X_test, y_train, y_test, X, y
        """
        X_train, X_test, y_train, y_test, X, y = get_data(
            self.n_points_per_group,
            return_full_data=True,
            filepath=DATA_FILEPATH
        )
        X_train, X_test, X = self._standardize_or_to_array("x", X_train, X_test, X)
        y_train, y_test, y = self._standardize_or_to_array("y", y_train, y_test, y)
        return X_train, X_test, y_train, y_test, X, y

    # todo: type hints!
    def compute_metrics(
        self, y_pred, y_quantiles, y_std, y_true, quantiles=None
    ):
        """

        :param y_pred: predicted y-values
        :param y_quantiles:
        :param y_std:
        :param y_true:
        :param quantiles:
        :return:
        """
        # todo: sharpness? calibration? PIT? coverage?
        # todo: skill score (but what to use as benchmark)?

        def clean_y(y):
            return np.array(y).squeeze()

        y_pred, y_quantiles, y_std, y_true = map(clean_y, (y_pred, y_quantiles, y_std, y_true))

        metrics = {  # todo: improve
            "rmse": rmse(y_true, y_pred),
            "smape_scaled": smape_scaled(y_true, y_pred),
            "crps": crps(y_true, y_pred, y_std),
            "nll_gaussian": nll_gaussian(y_true, y_pred, y_std),
            "mean_pinball": mean_pinball_loss(y_pred, y_quantiles, quantiles),
        }
        return metrics

    def train_base_model(self, *args, **kwargs):
        # todo: more flexibility in choosing (multiple) base models
        if TEMP_TEST_ALL:
            model = self.my_train_base_model_rf(*args, **kwargs)
            model = self.my_train_base_model_nn(*args, **kwargs)
        else:
            # model = self.my_train_base_model_rf(*args, **kwargs)
            model = self.my_train_base_model_nn(*args, **kwargs)
        return model

    def my_train_base_model_rf(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_params_choices=None,
        model_init_params=None,
        skip_training=True,
        n_jobs=-1,
        cv_n_iter=10,
        save_trained=True,
        verbose=True,
    ):
        # todo: more flexibility in choosing (multiple) base models
        if model_params_choices is None:
            model_params_choices = {
                "max_depth": randint(2, 30),
                "n_estimators": randint(10, 100),
            }
        random_seed = 42
        if model_init_params is None:
            model_init_params = {}
        elif "random_seed" not in model_init_params:
            model_init_params["random_seed"] = random_seed

        model_class = RandomForestRegressor
        filename_base_model = f"base_{model_class.__name__}.model"

        if skip_training:
            # Model previously optimized with a cross-validation:
            # RandomForestRegressor(max_depth=13, n_estimators=89, random_state=42)
            try:
                print('skipping base model training')
                model = self.io_helper.load_model(filename_base_model)
                return model
            except FileNotFoundError:
                print(f"trained base model '{filename_base_model}' not found")

        assert all(
            item is not None for item in [X_train, y_train, model_params_choices]
        )
        print("training random forest...")

        # CV parameter search
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        model = model_class(random_state=random_seed, **model_init_params)
        cv_obj = RandomizedSearchCV(
            model,
            param_distributions=model_params_choices,
            n_iter=cv_n_iter,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=random_seed,
            verbose=1,
            n_jobs=n_jobs,
        )
        # todo: ravel?
        cv_obj.fit(X_train, y_train.ravel())
        model = cv_obj.best_estimator_
        print("done")
        self.io_helper.save_model(model, filename_base_model)
        return model

    def my_train_base_model_nn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_params_choices=None,
        n_iter=500,
        batch_size=20,
        random_seed=42,
        verbose: int = 1,
        skip_training=True,
        save_trained=True,
        model_filename=None,
        val_frac=0.1,
        lr=0.1,
        lr_patience=5,
        lr_reduction_factor=0.5,
    ):
        """

        :param val_frac:
        :param lr_reduction_factor:
        :param lr:
        :param lr_patience:
        :param model_filename:
        :param save_trained:
        :param skip_training:
        :param verbose:
        :param X_train: shape (n_samples, n_dims)
        :param y_train: shape (n_samples, n_dims)
        :param model_params_choices:
        :param n_iter:
        :param batch_size:
        :param random_seed:
        :return:
        """

        X_train, y_train = map(numpy_to_tensor, (X_train, y_train))
        X_train, y_train = map(tensor_to_device, (X_train, y_train))

        if model_filename is None:
            n_training_points = X_train.shape[0]
            model_filename = f"base_nn_{n_training_points}.pth"
        if skip_training:
            print("skipping base model training")
            try:
                model = self.io_helper.load_torch_model(model_filename)
                model.eval()
                return model
            except FileNotFoundError:
                # fmt: off
                print("error. model not found, so training cannot be skipped. training from scratch")

        model = NN_Estimator(
            n_iter=n_iter,
            batch_size=batch_size,
            random_seed=random_seed,
            val_frac=val_frac,
            lr=lr,
            lr_patience=lr_patience,
            lr_reduction_factor=lr_reduction_factor,
            verbose=verbose,
        )
        model.fit(X_train, y_train)

        if save_trained:
            self.io_helper.save_torch_model(model, model_filename)

        # noinspection PyTypeChecker
        model.set_params(verbose=False)
        return model

    def posthoc_conformal_prediction(
        self,
        X_train,
        y_train,
        X_uq,
        quantiles,
        model,
        random_seed=42,
        n_estimators=10,
        bootstrap_n_blocks=10,
        bootstrap_overlapping_blocks=False,
        verbose=1,
        skip_training=False,
    ):
        """
        
        :param skip_training:
        :param verbose:
        :param X_train:
        :param y_train: 
        :param X_uq: 
        :param quantiles: 
        :param model: 
        :param random_seed: 
        :param n_estimators: number of model clones to train for ensemble
        :param bootstrap_n_blocks: 
        :param bootstrap_overlapping_blocks: 
        :return: 
        """
        cv = BlockBootstrap(
            n_resamplings=n_estimators,
            n_blocks=bootstrap_n_blocks,
            overlapping=bootstrap_overlapping_blocks,
            random_state=random_seed
        )
        alphas = self.pis_from_quantiles(quantiles)
        y_pred, y_pis = estimate_pred_interals_no_pfit_enbpi(
            model, cv, alphas, X_uq, X_train, y_train, skip_training=skip_training, io_helper=self.io_helper,
            agg_function='mean', verbose=verbose,
        )
        y_quantiles = self.quantiles_from_pis(y_pis)  # (n_samples, 2 * n_intervals)
        if 0.5 in quantiles:
            num_quantiles = y_quantiles.shape[-1]
            ind = num_quantiles / 2
            y_quantiles = np.insert(y_quantiles, ind, y_pred, axis=1)
        y_std = None  # self.stds_from_quantiles(y_quantiles)
        return y_pred, y_quantiles, y_std

    def posthoc_laplace(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_uq: np.ndarray,
        quantiles,
        model,
        n_iter=100,
        batch_size=20,
        random_seed=42,
        verbose=True,
    ):
        # todo: offer option to alternatively optimize parameters and hyperparameters of the prior jointly (cf. example
        #  script)?
        torch.manual_seed(random_seed)
        X_uq, X_train, y_train = map(numpy_to_tensor, (X_uq, X_train, y_train))
        X_uq, X_train, y_train = map(tensor_to_device, (X_uq, X_train, y_train))

        train_loader = get_train_loader(X_train, y_train, batch_size)

        la = Laplace(model, "regression")
        la.fit(train_loader)
        log_prior, log_sigma = (
            torch.ones(1, requires_grad=True),
            torch.ones(1, requires_grad=True),
        )
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
        iterable = tqdm(range(n_iter)) if verbose else range(n_iter)
        for _ in iterable:
            hyper_optimizer.zero_grad()
            neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
            neg_marglik.backward()
            hyper_optimizer.step()

        # # Serialization for fitted quantities
        # state_dict = la.state_dict()
        # torch.save(state_dict, "state_dict.bin")
        #
        # la = Laplace(model, "regression", subset_of_weights="all", hessian_structure="full")
        # # Load serialized, fitted quantities
        # la.load_state_dict(torch.load("state_dict.bin"))

        f_mu, f_var = la(X_uq)

        f_mu = tensor_to_numpy(f_mu.squeeze())
        f_sigma = tensor_to_numpy(f_var.squeeze().sqrt())
        pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)

        y_pred, y_std = f_mu, pred_std
        y_quantiles = self.quantiles_gaussian(quantiles, y_pred, y_std)
        return y_pred, y_quantiles, y_std

    def native_quantile_regression(self, X_train: np.ndarray, y_train: np.ndarray, X_uq: np.ndarray, quantiles, verbose=True):
        y_pred, y_quantiles = estimate_quantiles_qr(
            X_train, y_train, X_uq, alpha=quantiles
        )
        y_std = self.stds_from_quantiles(y_quantiles)
        return y_pred, y_quantiles, y_std

    @staticmethod
    def native_mvnn(X_train: np.ndarray, y_train: np.ndarray, X_uq: np.ndarray, quantiles, **kwargs):
        return run_mean_var_nn(
            X_train,
            y_train,
            X_uq,
            quantiles,
            **kwargs
        )

    def native_gp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_uq: np.ndarray,
        quantiles,
        verbose=True,
        n_restarts_optimizer=10,
    ):
        if verbose:
            print(f"fitting GP kernel... [{time.strftime('%H:%M:%S')}]")
        kernel = self._get_kernel()
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=42,
        )
        gaussian_process.fit(X_train, y_train)
        if verbose:
            print(f"done. [{time.strftime('%H:%M:%S')}]")
            print("kernel:", gaussian_process.kernel_)
            print("GP predicting...")
        mean_prediction, std_prediction = gaussian_process.predict(X_uq, return_std=True)
        if verbose:
            print("done.")
        y_pred, y_std = mean_prediction, std_prediction
        y_quantiles = self.quantiles_gaussian(quantiles, y_pred, y_std)
        return y_pred, y_quantiles, y_std

    @staticmethod
    def _get_kernel():
        return RBF() + WhiteKernel()

    @staticmethod
    def quantiles_gaussian(quantiles, y_pred, y_std):
        # todo: does this work for multi-dim outputs?
        return np.array([norm.ppf(quantiles, loc=mean, scale=std)
                         for mean, std in zip(y_pred, y_std)])

    def _standardize_or_to_array(self, variable, *dfs):
        if variable in self.to_standardize:
            return standardize(False, *dfs)
        return map(df_to_numpy, dfs)

    def plot_post_training_perf(self, base_model, X_train, y_train, do_save_figure=False, filename='base_model'):
        y_preds = base_model.predict(X_train)

        num_train_steps = X_train.shape[0]
        x_plot_train = np.arange(num_train_steps)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
        ax.plot(x_plot_train, y_train, label='y_train', linestyle="dashed", color="black")
        ax.plot(
            x_plot_train,
            y_preds,
            label=f"base model prediction",
            color="green",
        )
        ax.legend()
        ax.set_xlabel("data")
        ax.set_ylabel("target")
        if do_save_figure:
            self.io_helper.save_plot(f"{filename}.png")
        plt.show()


def print_metrics(uq_metrics: dict[str, dict[str, dict[str, Any]]]):
    print()
    for uq_type, method_metrics in uq_metrics.items():
        print(f"{uq_type} metrics:")
        for method, metrics in method_metrics.items():
            print(f"\t{method}:")
            for metric, value in metrics.items():
                print(f"\t\t{metric}: {value}")
        print()


def main():
    uq_comparer = My_UQ_Comparer(
        methods_kwargs=METHODS_KWARGS,
        method_whitelist=METHOD_WHITELIST,
        to_standardize=TO_STANDARDIZE,
        n_points_per_group=N_POINTS_PER_GROUP,
    )
    uq_metrics = uq_comparer.compare_methods(
        QUANTILES,
        should_plot_data=PLOT_DATA,
        should_plot_results=PLOT_RESULTS,
        should_show_plots=SHOW_PLOTS,
        should_save_plots=SAVE_PLOTS,
        plots_path=PLOTS_PATH,
        output_uq_on_train=True,
        return_results=False,
    )
    print_metrics(uq_metrics)


def temp_test():
    uq_comparer = My_UQ_Comparer(
        method_whitelist=METHOD_WHITELIST, to_standardize=TO_STANDARDIZE
    )
    X_train, X_test, y_train, y_test, X, y = uq_comparer.get_data()
    filename_enbpi_no_pfit = f"mapie_enbpi_no_pfit_800_1600.model"
    mapie_enbpi = uq_comparer.io_helper.load_model(filename_enbpi_no_pfit)

    multi_y = [e.predict(X) for e in mapie_enbpi.estimator_.estimators_]
    x_plot_full = np.arange(X.shape[0])
    x_plot_train = np.arange(X_train.shape[0])
    x_plot_test = np.arange(X_test.shape[0], len(x_plot_train) + X_test.shape[0])
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, y_i in enumerate(multi_y):
        ax.plot(x_plot_full, y_i, alpha=0.5, label=f'y_pred{i}')
    ax.plot(x_plot_train, y_train, color='black', linestyle="dashed", label='y_train')
    ax.plot(x_plot_test, y_test, color='black', label='y_test')
    ax.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
