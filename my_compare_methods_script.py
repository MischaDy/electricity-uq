import os

import pandas as pd

filename = os.path.split(__file__)[-1]
print(f'reading file {filename}...')

import numpy as np
from scipy import stats

from compare_methods import UQ_Comparer
from helpers import get_data, standardize, train_val_split
from nn_estimator import NN_Estimator


METHOD_WHITELIST = [
    # "posthoc_conformal_prediction",
    # "posthoc_laplace",
    # "native_quantile_regression",
    # "native_gpytorch",
    # "native_mvnn",
    # 'base_model_rf',
     'base_model_nn',
]
POSTHOC_BASE_BLACKLIST = {
    'posthoc_laplace': {
        'base_model_rf',
    },
}
QUANTILES = [0.05, 0.25, 0.75, 0.95]  # todo: how to handle 0.5? ==> just use mean if needed

DATA_FILEPATH = './data_1600.pkl'

N_POINTS_PER_GROUP = 800
PLOT_DATA = False
PLOT_UQ_RESULTS = True
PLOT_BASE_RESULTS = True
SHOW_PLOTS = False
SAVE_PLOTS = True
SKIP_BASE_MODEL_COPY = True

METHODS_KWARGS = {
    "native_mvnn": {
        "n_iter": 300,
        "lr": 1e-4,
        "lr_patience": 30,
        "regularization": 0,  # 1e-2,
        "warmup_period": 50,
        "frozen_var_value": 0.1,
        'skip_training': True,
        'save_model': True,
    },
    "native_quantile_regression": {
        'skip_training': True,
        'save_model': True,
        "verbose": True,
    },
    "native_gpytorch": {
        'n_epochs': 100,
        'val_frac': 0.1,
        'lr': 1e-2,
        'show_progress': True,
        'show_plots': True,
        'do_plot_losses': True,
        'skip_training': True,
        'save_model': True,
        'model_name': 'gpytorch_model',
        'verbose': True,
    },
    "posthoc_conformal_prediction": {
        "n_estimators": 5,
        "verbose": 1,
        "skip_training": True,
        "save_model": True,
    },
    "posthoc_laplace": {
        "n_iter": 100,
        'skip_training': True,
        'save_model': True,
    },
    "base_model_nn": {
        "n_iter": 200,
        "lr": 1e-2,
        "lr_patience": 30,
        "lr_reduction_factor": 0.5,
        "show_progress_bar": True,
        "show_losses_plot": False,
        "save_losses_plot": True,
        "random_seed": 42,
        "skip_training": False,
        "save_model": True,
        "verbose": 1,
    },
    "base_model_rf": {
        'model_param_distributions': {
            "max_depth": stats.randint(2, 50),
            "n_estimators": stats.randint(10, 200),
        },
        'cv_n_iter': 20,
        'n_cv_splits': 5,
        "random_seed": 42,
        "skip_training": True,
        "save_model": True,
        "verbose": 4,
        'n_jobs': -1,
    },
}

TO_STANDARDIZE = "xy"


# noinspection PyPep8Naming
class My_UQ_Comparer(UQ_Comparer):
    def __init__(
            self,
            storage_path="comparison_storage",
            to_standardize="X",
            methods_kwargs=None,  # : dict[str, dict[str, Any]] = None,
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
        super().__init__(*args, storage_path=storage_path, **kwargs)
        if methods_kwargs is not None:
            self.methods_kwargs.update(methods_kwargs)
        self.to_standardize = to_standardize
        self.n_points_per_group = n_points_per_group

    def get_data(self):
        """
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

    @classmethod
    def compute_metrics_det(cls, y_pred, y_true) -> dict[str, float]:
        # todo: sharpness? calibration? PIT? coverage?
        from metrics import rmse, smape_scaled

        y_pred, y_true = cls._clean_ys_for_metrics(y_pred, y_true)
        metrics = {
            "rmse": rmse(y_true, y_pred),
            "smape_scaled": smape_scaled(y_true, y_pred),
        }
        return cls._clean_metrics(metrics)

    @classmethod
    def compute_metrics_uq(cls, y_pred, y_quantiles, y_std, y_true, quantiles=None) -> dict[str, float]:
        # todo: sharpness? calibration? PIT? coverage?
        from metrics import crps, nll_gaussian, mean_pinball_loss

        y_pred, y_quantiles, y_std, y_true = cls._clean_ys_for_metrics(y_pred, y_quantiles, y_std, y_true)
        metrics = {
            "crps": crps(y_true, y_pred, y_std),
            "nll_gaussian": nll_gaussian(y_true, y_pred, y_std),
            "mean_pinball": mean_pinball_loss(y_pred, y_quantiles, quantiles),
        }
        return cls._clean_metrics(metrics)

    @staticmethod
    def _clean_ys_for_metrics(*ys):
        for y in ys:
            if y is None:
                yield y
            else:
                yield np.array(y).squeeze()

    @staticmethod
    def _clean_metrics(metrics):
        metrics = {
            metric_name: (None if value is None else float(value))
            for metric_name, value in metrics.items()
        }
        return metrics

    def base_model_rf(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            model_param_distributions=None,
            n_jobs=-1,
            cv_n_iter=100,
            n_cv_splits=10,
            random_seed=42,
            skip_training=True,
            save_model=True,
            verbose=1,
    ):
        """

        :param random_seed:
        :param n_cv_splits:
        :param X_train:
        :param y_train:
        :param model_param_distributions:
        :param skip_training:
        :param n_jobs:
        :param cv_n_iter:
        :param save_model:
        :param verbose:
        :return:
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

        # todo: more flexibility in choosing (multiple) base models
        if model_param_distributions is None:
            model_param_distributions = {
                "max_depth": stats.randint(2, 100),
                "n_estimators": stats.randint(10, 1000),
            }

        model_class = RandomForestRegressor
        filename_base_model = f"base_{model_class.__name__}.model"

        if skip_training:
            try:
                print('skipping base model training')
                model = self.io_helper.load_model(filename_base_model)
                return model
            except FileNotFoundError:
                print(f"trained base model '{filename_base_model}' not found. training from scratch.")

        assert all(item is not None for item in [X_train, y_train, model_param_distributions])
        print("training random forest...")

        # CV parameter search
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)
        model = model_class(random_state=random_seed)
        cv_obj = RandomizedSearchCV(
            model,
            param_distributions=model_param_distributions,
            n_iter=cv_n_iter,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        # todo: ravel?
        cv_obj.fit(X_train, y_train.ravel())
        model = cv_obj.best_estimator_
        print("done.")
        if save_model:
            print('saving model...')
            self.io_helper.save_model(model, filename_base_model)
        return model

    def base_model_nn(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            n_iter=500,
            batch_size=20,
            random_seed=42,
            model_filename=None,
            val_frac=0.1,
            lr=0.1,
            lr_patience=30,
            lr_reduction_factor=0.5,
            show_progress_bar=True,
            show_losses_plot=True,
            save_losses_plot=True,
            skip_training=True,
            save_model=True,
            verbose: int = 1,
    ):
        """

        :param show_losses_plot:
        :param save_losses_plot:
        :param show_progress_bar:
        :param val_frac:
        :param lr_reduction_factor:
        :param lr:
        :param lr_patience:
        :param model_filename:
        :param save_model:
        :param skip_training:
        :param verbose:
        :param X_train: shape (n_samples, n_dims)
        :param y_train: shape (n_samples, n_dims)
        :param n_iter:
        :param batch_size:
        :param random_seed:
        :return:
        """
        from nn_estimator import NN_Estimator
        from helpers import object_to_cuda

        if model_filename is None:
            n_training_points = X_train.shape[0]
            model_filename = f"base_nn_{n_training_points}.pth"
        if skip_training:
            print("skipping base model training")
            try:
                model = self.io_helper.load_torch_model(model_filename)
                model = object_to_cuda(model)
                model.eval()
                return model
            except FileNotFoundError:
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
            show_progress_bar=show_progress_bar,
            save_losses_plot=save_losses_plot,
            show_losses_plot=show_losses_plot,
        )
        # noinspection PyTypeChecker
        model.fit(X_train, y_train)

        if save_model:
            print('saving model...')
            self.io_helper.save_torch_model(model, model_filename)

        # noinspection PyTypeChecker
        model.set_params(verbose=False)
        return model

    def posthoc_conformal_prediction(
            self,
            X_train,
            y_train,
            X_pred,
            quantiles,
            model,
            random_seed=42,
            n_estimators=10,
            bootstrap_n_blocks=10,
            bootstrap_overlapping_blocks=False,
            verbose=1,
            skip_training=True,
            save_model=True,
    ):
        """
        
        :param save_model:
        :param skip_training:
        :param verbose:
        :param X_train:
        :param y_train: 
        :param X_pred:
        :param quantiles: 
        :param model: 
        :param random_seed: 
        :param n_estimators: number of model clones to train for ensemble
        :param bootstrap_n_blocks: 
        :param bootstrap_overlapping_blocks: 
        :return: 
        """
        from conformal_prediction import estimate_pred_interals_no_pfit_enbpi
        from mapie.subsample import BlockBootstrap

        cv = BlockBootstrap(
            n_resamplings=n_estimators,
            n_blocks=bootstrap_n_blocks,
            overlapping=bootstrap_overlapping_blocks,
            random_state=random_seed,
        )
        alphas = self.pis_from_quantiles(quantiles)
        y_pred, y_pis = estimate_pred_interals_no_pfit_enbpi(
            model,
            cv,
            alphas,
            X_pred,
            X_train,
            y_train,
            skip_training=skip_training,
            save_model=save_model,
            io_helper=self.io_helper,
            agg_function='mean',
            verbose=verbose,
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
            X_pred: np.ndarray,
            quantiles,
            base_model: NN_Estimator,
            n_iter=100,
            batch_size=20,
            random_seed=42,
            skip_training=True,
            model_filename=None,
            save_model=True,
            verbose=True,
    ):
        # todo: offer option to alternatively optimize parameters and hyperparameters of the prior jointly (cf. example
        #  script)?
        from laplace import Laplace
        from tqdm import tqdm
        from helpers import (get_train_loader, tensor_to_np_array, np_arrays_to_tensors, np_array_to_tensor,
                             objects_to_cuda, object_to_cuda, make_tensors_contiguous, make_tensor_contiguous,
                             get_device)
        import torch
        from torch import nn

        torch.set_default_dtype(torch.float32)
        torch.manual_seed(random_seed)
        torch.set_default_device(get_device())

        def la_instantiator(base_model: nn.Module):
            return Laplace(base_model, "regression")

        n_training_points = X_train.shape[0]
        if model_filename is None:
            model_filename = f"laplace_{n_training_points}_{n_iter}.pth"

        base_model_nn = base_model.get_nn(to_device=True)
        if skip_training:
            print("skipping model training...")
            try:
                # noinspection PyTypeChecker
                la = self.io_helper.load_laplace_model_statedict(
                    base_model_nn,
                    la_instantiator,
                    laplace_model_filename=model_filename,
                )
            except FileNotFoundError:
                print(f"error. model {model_filename} not found, so training cannot be skipped. training from scratch.")
                skip_training = False

        if not skip_training:
            X_train, y_train = np_arrays_to_tensors(X_train, y_train)
            X_train, y_train = objects_to_cuda(X_train, y_train)
            X_train, y_train = make_tensors_contiguous(X_train, y_train)

            train_loader = get_train_loader(X_train, y_train, batch_size)
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

        if save_model:
            if skip_training:
                print('skipped training, so not saving model.')
            else:
                print('saving model...')
                # noinspection PyUnboundLocalVariable
                self.io_helper.save_laplace_model_statedict(
                    la,
                    laplace_model_filename=model_filename
                )

        print('predicting...')

        # import ipdb
        # ipdb.set_trace()

        X_pred = np_array_to_tensor(X_pred)
        X_pred = object_to_cuda(X_pred)
        X_pred = make_tensor_contiguous(X_pred)

        # noinspection PyArgumentList
        f_mu, f_var = la(X_pred)
        f_mu = tensor_to_np_array(f_mu.squeeze())
        f_sigma = tensor_to_np_array(f_var.squeeze().sqrt())
        pred_std = np.sqrt(f_sigma ** 2 + la.sigma_noise.item() ** 2)

        y_pred, y_std = f_mu, pred_std
        y_quantiles = self.quantiles_gaussian(quantiles, y_pred, y_std)
        return y_pred, y_quantiles, y_std

    def native_quantile_regression(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_pred: np.ndarray,
            quantiles,
            verbose=True,
            skip_training=True,
            save_model=True,
    ):
        from quantile_regression import estimate_quantiles as estimate_quantiles_qr
        y_pred, y_quantiles = estimate_quantiles_qr(
            X_train,
            y_train,
            X_pred,
            alpha=quantiles,
            skip_training=skip_training,
            save_model=save_model,
            verbose=verbose,
        )
        y_std = self.stds_from_quantiles(y_quantiles)
        return y_pred, y_quantiles, y_std

    @staticmethod
    def native_mvnn(
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_pred: np.ndarray,
            quantiles,
            n_iter=300,
            lr=1e-4,
            lr_patience=30,
            regularization=0,  # 1e-2,
            warmup_period=50,
            frozen_var_value=0.1,
            skip_training=True,
            save_model=True,
    ):
        from mean_var_nn import run_mean_var_nn
        return run_mean_var_nn(
            X_train,
            y_train,
            X_pred,
            quantiles,
            n_iter=n_iter,
            lr=lr,
            lr_patience=lr_patience,
            regularization=regularization,
            warmup_period=warmup_period,
            frozen_var_value=frozen_var_value,
            do_plot_losses=False,
            use_scheduler=True,
            skip_training=skip_training,
            save_model=save_model,
        )

    def native_gpytorch(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_pred: np.ndarray,
            quantiles,
            n_epochs=100,
            val_frac=0.1,
            model_name='gpytorch_model',
            lr=1e-2,
            show_progress=True,
            show_plots=True,
            do_plot_losses=True,
            skip_training=True,
            save_model=True,
            verbose=True,
    ):
        import torch
        import gpytorch
        from gp_regression_gpytorch import ExactGPModel, train_gpytorch
        from helpers import (make_ys_1d, np_arrays_to_tensors, make_tensors_contiguous, objects_to_cuda,
                             tensors_to_np_arrays)

        print('preparing data..')
        X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_frac)
        X_train, y_train, X_val, y_val, X_pred = np_arrays_to_tensors(X_train, y_train, X_val, y_val, X_pred)
        y_train, y_val = make_ys_1d(y_train, y_val)

        print('mapping data to device and making it contiguous...')
        X_train, y_train, X_val, y_val, X_pred = objects_to_cuda(X_train, y_train, X_val, y_val, X_pred)
        X_train, y_train, X_val, y_val, X_pred = make_tensors_contiguous(X_train, y_train, X_val, y_val, X_pred)

        common_prefix, common_postfix = f'{model_name}', f'{self.n_points_per_group}_{n_epochs}'
        model_name = f'{common_prefix}_{common_postfix}.pth'
        model_likelihood_name = f'{common_prefix}_likelihood_{common_postfix}.pth'
        if skip_training:
            print('skipping training...')
            try:
                likelihood = self.io_helper.load_torch_model_statedict(gpytorch.likelihoods.GaussianLikelihood,
                                                                       model_likelihood_name)
                model = self.io_helper.load_torch_model_statedict(ExactGPModel, model_name,
                                                                  X_train=X_train, y_train=y_train,
                                                                  likelihood=likelihood)
                model, likelihood = objects_to_cuda(model, likelihood)
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
                n_epochs,
                lr=lr,
                show_progress=show_progress,
                show_plots=show_plots,
                do_plot_losses=do_plot_losses,
            )

        # noinspection PyUnboundLocalVariable
        model.eval()
        # noinspection PyUnboundLocalVariable
        likelihood.eval()

        if save_model:
            if skip_training:
                print('skipped training, so not saving models.')
            else:
                print('saving...')
                self.io_helper.save_torch_model_statedict(model, model_name)
                self.io_helper.save_torch_model_statedict(likelihood, model_likelihood_name)

        with torch.no_grad():  # todo: use gpytorch.settings.fast_pred_var()?
            f_preds = model(X_pred)
        y_preds = f_preds.mean
        y_std = f_preds.stddev
        y_preds, y_std = tensors_to_np_arrays(y_preds, y_std)

        y_quantiles = self.quantiles_gaussian(quantiles, y_preds, y_std)
        return y_preds, y_quantiles, y_std

    @staticmethod
    def quantiles_gaussian(quantiles, y_pred: np.ndarray, y_std: np.ndarray):
        from scipy.stats import norm
        # todo: does this work for multi-dim outputs?
        return np.array([norm.ppf(quantiles, loc=mean, scale=std)
                         for mean, std in zip(y_pred, y_std)])

    def _standardize_or_to_array(self, variable, *dfs: pd.DataFrame):
        from helpers import dfs_to_np_arrays

        if variable in self.to_standardize:
            return standardize(*dfs, return_scaler=False)
        return dfs_to_np_arrays(*dfs)

    def plot_base_model_test_result(
            self,
            X_train,
            X_test,
            y_train,
            y_test,
            y_preds,
            plot_name='base_model',
            show_plots=True,
            save_plot=True,
    ):
        from matplotlib import pyplot as plt
        num_train_steps, num_test_steps = X_train.shape[0], X_test.shape[0]

        x_plot_train = np.arange(num_train_steps)
        x_plot_full = np.arange(num_train_steps + num_test_steps)
        x_plot_test = np.arange(num_train_steps, num_train_steps + num_test_steps)
        x_plot_uq = x_plot_full

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
        ax.plot(x_plot_train, y_train, label='y_train', linestyle="dashed", color="black")
        ax.plot(x_plot_test, y_test, label='y_test', linestyle="dashed", color="blue")
        ax.plot(
            x_plot_uq,
            y_preds,
            label=f"base model prediction {plot_name}",
            color="green",
        )
        ax.legend()
        ax.set_xlabel("data")
        ax.set_ylabel("target")
        ax.set_title(plot_name)
        if save_plot:
            self.io_helper.save_plot(plot_name)
        if show_plots:
            plt.show(block=True)
        plt.close(fig)


def main():
    import torch
    torch.set_default_dtype(torch.float32)

    uq_comparer = My_UQ_Comparer(
        methods_kwargs=METHODS_KWARGS,
        method_whitelist=METHOD_WHITELIST,
        posthoc_base_blacklist=POSTHOC_BASE_BLACKLIST,
        to_standardize=TO_STANDARDIZE,
        n_points_per_group=N_POINTS_PER_GROUP,
    )
    uq_comparer.compare_methods(
        QUANTILES,
        should_plot_data=PLOT_DATA,
        should_plot_uq_results=PLOT_UQ_RESULTS,
        should_plot_base_results=PLOT_BASE_RESULTS,
        should_show_plots=SHOW_PLOTS,
        should_save_plots=SAVE_PLOTS,
        skip_base_model_copy=SKIP_BASE_MODEL_COPY,
    )


if __name__ == "__main__":
    main()
