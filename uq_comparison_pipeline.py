import logging
import os
from typing import Any, Generator

filename = os.path.split(__file__)[-1]
logging.info(f'reading file {filename}...')

import numpy as np
from scipy import stats

from uq_comparison_pipeline_abc import UQ_Comparison_Pipeline_ABC
from helpers.misc_helpers import get_data, train_val_split, preprocess_array
from src_base_models.nn_estimator import NN_Estimator


QUANTILES = [0.05, 0.25, 0.75, 0.95]  # todo: how to handle 0.5? ==> just use mean if needed

DATA_FILEPATH = 'data/data_1600.pkl'

N_POINTS_PER_GROUP = 800
STANDARDIZE_DATA = True

PLOT_DATA = False
PLOT_UQ_RESULTS = True
PLOT_BASE_RESULTS = True
SHOW_PLOTS = True
SAVE_PLOTS = True
SKIP_BASE_MODEL_COPY = True
SHOULD_SAVE_RESULTS = True

DO_TRAIN_ALL = True
SKIP_TRAINING_ALL = False

STORAGE_PATH = "comparison_storage"

METHOD_WHITELIST = [
    # "posthoc_conformal_prediction",
    # "posthoc_laplace",
    # "native_quantile_regression",
    "native_quantile_regression_nn",
    # "native_gpytorch",
    # "native_mvnn",
    # 'base_model_linreg',
    # 'base_model_rf',
    # 'base_model_nn',
]
POSTHOC_BASE_BLACKLIST = {
    'posthoc_laplace': {
        'base_model_linreg',
        'base_model_rf',
    },
}

METHODS_KWARGS = {
    "native_mvnn": {
        'skip_training': False,
        "num_hidden_layers": 2,
        "hidden_layer_size": 50,
        "activation": None,  # defaults to leaky ReLU
        "n_iter": 100,
        "lr": 1e-4,
        "lr_patience": 30,
        "regularization": 0,  # 1e-2,
        "warmup_period": 50,
        "frozen_var_value": 0.1,
        'save_model': True,
    },
    "native_quantile_regression": {
        'skip_training': False,
        'save_model': True,
        "verbose": True,
    },
    "native_quantile_regression_nn": {
        'n_iter': 300,
        'num_hidden_layers': 2,
        'hidden_layer_size': 50,
        'activation': None,
        'random_seed': 42,
        'lr': 1e-4,
        'use_scheduler': True,
        'lr_patience': 30,
        'regularization': 0,
        'show_progress': True,
        'do_plot_losses': True,
        'skip_training': True,
        'save_model': True,
    },
    "native_gpytorch": {
        'skip_training': False,
        'n_iter': 500,
        'val_frac': 0.1,
        'lr': 1e-2,
        'use_scheduler': True,
        'lr_patience': 30,
        'lr_reduction_factor': 0.5,
        'show_progress': True,
        'show_plots': True,
        'do_plot_losses': True,
        'save_model': True,
    },
    "posthoc_conformal_prediction": {
        "skip_training": False,
        "n_estimators": 5,
        "verbose": 1,
        "save_model": True,
    },
    "posthoc_laplace": {
        'skip_training': False,
        "n_iter": 100,
        'save_model': True,
    },
    "base_model_linreg": {
        "skip_training": True,
        "n_jobs": -1,
        "save_model": True,
    },
    "base_model_nn": {
        "skip_training": False,
        "n_iter": 100,
        "lr": 1e-2,
        "lr_patience": 30,
        "lr_reduction_factor": 0.5,
        "show_progress_bar": True,
        "show_losses_plot": False,
        "save_losses_plot": True,
        "random_seed": 42,
        "save_model": True,
        "verbose": 1,
    },
    "base_model_rf": {
        "skip_training": False,
        'model_param_distributions': {
            "max_depth": stats.randint(2, 50),
            "n_estimators": stats.randint(10, 200),
        },
        'cv_n_iter': 20,
        'cv_n_splits': 5,
        "random_seed": 42,
        "save_model": True,
        "verbose": 4,
        'n_jobs': -1,
    },
}

assert not (DO_TRAIN_ALL and SKIP_TRAINING_ALL)

for _, method_kwargs in METHODS_KWARGS.items():
    if DO_TRAIN_ALL:
        method_kwargs['skip_training'] = False
    elif SKIP_TRAINING_ALL:
        method_kwargs['skip_training'] = True


# noinspection PyPep8Naming
class UQ_Comparison_Pipeline(UQ_Comparison_Pipeline_ABC):
    def __init__(
            self,
            *,
            storage_path="comparison_storage",
            methods_kwargs: dict[str, dict[str, Any]] = None,
            n_points_per_group=800,
            method_whitelist=None,
            posthoc_base_blacklist: dict[str, set[str]] = None,
            standardize_data=True,
    ):
        """

        :param methods_kwargs: dict of (method_name, method_kwargs) pairs
        :param storage_path:
        :param n_points_per_group: both training size and test size
        :param standardize_data: True if both X and y should be standardized, False if neither.
        """
        super().__init__(storage_path=storage_path, method_whitelist=method_whitelist,
                         posthoc_base_blacklist=posthoc_base_blacklist, standardize_data=standardize_data)
        if methods_kwargs is None:
            methods_kwargs = {}
        self.methods_kwargs.update(methods_kwargs)
        self.n_points_per_group = n_points_per_group

    def get_data(self):
        """
        load and prepare data

        :return:
        A tuple (X_train, X_test, y_train, y_test, X, y, y_scaler). If self.standardize_data=False, y_scaler is None.
        All variables except for the scaler are 2D np arrays.
        """
        return get_data(
            filepath=DATA_FILEPATH,
            n_points_per_group=self.n_points_per_group,
            standardize_data=self.standardize_data,
        )

    @classmethod
    def compute_metrics_det(cls, y_pred, y_true) -> dict[str, float]:
        # todo: sharpness? calibration? PIT? coverage?
        from helpers.metrics import rmse, smape_scaled

        y_pred, y_true = cls._clean_ys_for_metrics(y_pred, y_true)
        metrics = {
            "rmse": rmse(y_true, y_pred),
            "smape_scaled": smape_scaled(y_true, y_pred),
        }
        return cls._clean_metrics(metrics)

    @classmethod
    def compute_metrics_uq(cls, y_pred, y_quantiles, y_std, y_true, quantiles=None) -> dict[str, float]:
        # todo: sharpness? calibration? PIT? coverage?
        from helpers.metrics import crps, nll_gaussian, mean_pinball_loss

        y_pred, y_quantiles, y_std, y_true = cls._clean_ys_for_metrics(y_pred, y_quantiles, y_std, y_true)
        metrics = {
            "crps": crps(y_true, y_quantiles),
            "nll_gaussian": nll_gaussian(y_true, y_pred, y_std),
            "mean_pinball": mean_pinball_loss(y_pred, y_quantiles, quantiles),
        }
        return cls._clean_metrics(metrics)

    def base_model_linreg(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            n_jobs=-1,
            skip_training=True,
            save_model=True,
    ):
        from sklearn import linear_model

        n_samples = X_train.shape[0]
        filename_base_model = f"base_model_linreg_n{n_samples}.model"
        if skip_training:
            try:
                logging.info('skipping linreg base model training')
                model = self.io_helper.load_model(filename_base_model)
                return model
            except FileNotFoundError:
                logging.warning(f"trained base model '{filename_base_model}' not found. training from scratch.")
        model = linear_model.LinearRegression(n_jobs=n_jobs)
        model.fit(X_train, y_train)
        if save_model:
            logging.info('saving linreg base model...')
            self.io_helper.save_model(model, filename_base_model)
        return model

    def base_model_rf(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            model_param_distributions=None,
            cv_n_iter=100,
            cv_n_splits=10,
            n_jobs=-1,
            random_seed=42,
            skip_training=True,
            save_model=True,
            verbose=1,
    ):
        """

        :param random_seed:
        :param cv_n_splits:
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

        n_samples = X_train.shape[0]
        filename_base_model = f"base_model_rf_n{n_samples}_it{cv_n_iter}_its{cv_n_splits}.model"

        if skip_training:
            try:
                logging.info('skipping random forest base model training')
                model = self.io_helper.load_model(filename_base_model)
                return model
            except FileNotFoundError:
                logging.warning(f"trained base model '{filename_base_model}' not found. training from scratch.")

        assert all(item is not None for item in [X_train, y_train, model_param_distributions])
        logging.info("training random forest...")

        # CV parameter search
        model = RandomForestRegressor(random_state=random_seed)
        cv_obj = RandomizedSearchCV(
            model,
            param_distributions=model_param_distributions,
            n_iter=cv_n_iter,
            cv=TimeSeriesSplit(n_splits=cv_n_splits),
            scoring="neg_root_mean_squared_error",
            random_state=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        # todo: ravel?
        cv_obj.fit(X_train, y_train.ravel())
        model = cv_obj.best_estimator_
        logging.info("done.")
        if save_model:
            logging.info('saving RF base model...')
            self.io_helper.save_model(model, filename_base_model)
        return model

    def base_model_nn(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            n_iter=500,
            batch_size=20,
            random_seed=42,
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
        from src_base_models.nn_estimator import NN_Estimator
        from helpers.misc_helpers import object_to_cuda

        n_samples = X_train.shape[0]
        model_filename = f"base_model_nn_n{n_samples}_it{n_iter}.pth"
        if skip_training:
            logging.info("skipping NN base model training")
            try:
                model = self.io_helper.load_torch_model(model_filename)
                model = object_to_cuda(model)
                model.eval()
                return model
            except FileNotFoundError:
                logging.warning("model not found, so training cannot be skipped. training from scratch")

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
            logging.info('saving NN base model...')
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
            n_estimators=10,
            bootstrap_n_blocks=10,
            bootstrap_overlapping_blocks=False,
            random_seed=42,
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
        from src_uq_methods_posthoc.conformal_prediction import estimate_pred_interals_no_pfit_enbpi
        from mapie.subsample import BlockBootstrap

        cv = BlockBootstrap(
            n_resamplings=n_estimators,
            n_blocks=bootstrap_n_blocks,
            overlapping=bootstrap_overlapping_blocks,
            random_state=random_seed,
        )
        alphas = self.pis_from_quantiles(quantiles)

        n_samples = X_train.shape[0]
        filename = f'posthoc_conformal_prediction_n{n_samples}_it{n_estimators}.model'
        y_pred, y_pis = estimate_pred_interals_no_pfit_enbpi(
            model,
            cv,
            alphas,
            X_pred,
            X_train,
            y_train,
            filename=filename,
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
            save_model=True,
            verbose=True,
    ):
        # todo: offer option to alternatively optimize parameters and hyperparameters of the prior jointly (cf. example
        #  script)?
        from laplace import Laplace
        from tqdm import tqdm
        from helpers import misc_helpers
        import torch
        from torch import nn

        torch.set_default_dtype(torch.float32)
        torch.manual_seed(random_seed)
        torch.set_default_device(misc_helpers.get_device())

        def la_instantiator(base_model: nn.Module):
            return Laplace(base_model, "regression")

        n_samples = X_train.shape[0]
        model_filename = f"posthoc_laplace_n{n_samples}_it{n_iter}.pth"
        base_model_nn = base_model.get_nn(to_device=True)
        if skip_training:
            logging.info("skipping model training...")
            try:
                # noinspection PyTypeChecker
                la = self.io_helper.load_laplace_model_statedict(
                    base_model_nn,
                    la_instantiator,
                    laplace_model_filename=model_filename,
                )
            except FileNotFoundError:
                logging.warning(f"model {model_filename} not found, so training cannot be skipped. training from scratch.")
                skip_training = False

        if not skip_training:
            X_train, y_train = misc_helpers.np_arrays_to_tensors(X_train, y_train)
            X_train, y_train = misc_helpers.objects_to_cuda(X_train, y_train)
            X_train, y_train = misc_helpers.make_tensors_contiguous(X_train, y_train)

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

            if save_model:
                logging.info('saving model...')
                self.io_helper.save_laplace_model_statedict(
                    la,
                    laplace_model_filename=model_filename
                )

        logging.info('predicting...')

        X_pred = misc_helpers.np_array_to_tensor(X_pred)
        X_pred = misc_helpers.object_to_cuda(X_pred)
        X_pred = misc_helpers.make_tensor_contiguous(X_pred)

        # noinspection PyArgumentList,PyUnboundLocalVariable
        f_mu, f_var = la(X_pred)
        f_mu = misc_helpers.tensor_to_np_array(f_mu.squeeze())
        f_sigma = misc_helpers.tensor_to_np_array(f_var.squeeze().sqrt())
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
        from src_uq_methods_native.quantile_regression import estimate_quantiles as estimate_quantiles_qr
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

    def native_quantile_regression_nn(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_pred: np.ndarray,
            quantiles: list,
            n_iter=300,
            num_hidden_layers=2,
            hidden_layer_size=50,
            activation=None,
            random_seed=42,
            lr=1e-4,
            use_scheduler=True,
            lr_patience=30,
            regularization=0,
            show_progress=True,
            do_plot_losses=True,
            skip_training=True,
            save_model=True,
    ):
        import torch
        from helpers import misc_helpers
        from src_uq_methods_native.quantile_regression_nn import QR_NN, train_qr_nn

        torch.manual_seed(random_seed)

        if activation is None:
            activation = torch.nn.LeakyReLU

        if 0.5 not in quantiles:
            quantiles.append(0.5)

        n_samples = X_train.shape[0]
        filename = f'native_qrnn_n{n_samples}_it{n_iter}_nh{num_hidden_layers}_hs{hidden_layer_size}.pth'
        if skip_training:
            logging.info('skipping training...')
            try:
                model = self.io_helper.load_torch_model_statedict(
                    QR_NN,
                    filename,
                    dim_in=X_train.shape[0],
                    num_hidden_layers=num_hidden_layers,
                    hidden_layer_size=hidden_layer_size,
                    activation=activation,
                )
                model = misc_helpers.objects_to_cuda(model)
            except FileNotFoundError:
                logging.warning(f'cannot load model {filename}')
                skip_training = False

        if not skip_training:
            logging.info('training from scratch...')
            model = train_qr_nn(
                X_train,
                y_train,
                quantiles,
                n_iter=n_iter,
                lr=lr,
                use_scheduler=use_scheduler,
                lr_patience=lr_patience,
                weight_decay=regularization,
                do_plot_losses=do_plot_losses,
                show_progress=show_progress,
            )
            if save_model:
                logging.info('saving model...')
                model.eval()
                self.io_helper.save_torch_model_statedict(model, filename)

        X_pred = preprocess_array(X_pred)
        with torch.no_grad():
            # noinspection PyUnboundLocalVariable,PyCallingNonCallable
            y_quantiles_dict = model(X_pred, as_dict=True)
        y_quantiles = np.array(list(y_quantiles_dict.values())).T
        y_pred = y_quantiles_dict[0.5]
        y_std = None
        return y_pred, y_quantiles, y_std

    def native_mvnn(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_pred: np.ndarray,
            quantiles,
            n_iter=300,
            num_hidden_layers=2,
            hidden_layer_size=50,
            activation=None,
            lr=1e-4,
            lr_patience=30,
            regularization=0,  # 1e-2,
            warmup_period=50,
            frozen_var_value=0.1,
            skip_training=True,
            save_model=True,
    ):
        import torch
        from helpers import misc_helpers
        from src_uq_methods_native.mean_var_nn import MeanVarNN, train_mean_var_nn

        if activation is None:
            activation = torch.nn.LeakyReLU

        n_samples = X_train.shape[0]
        filename = f'native_mvnn_n{n_samples}_it{n_iter}_nh{num_hidden_layers}_hs{hidden_layer_size}.pth'
        if skip_training:
            logging.info('skipping training...')
            try:
                model = self.io_helper.load_torch_model_statedict(
                    MeanVarNN,
                    filename,
                    dim_in=X_train.shape[0],
                    num_hidden_layers=num_hidden_layers,
                    hidden_layer_size=hidden_layer_size,
                    activation=activation,
                )
                model = misc_helpers.objects_to_cuda(model)
            except FileNotFoundError:
                logging.warning(f'cannot load model {filename}')
                skip_training = False

        if not skip_training:
            logging.info('training from scratch...')

            common_params = {
                "X_train": X_train,
                "y_train": y_train,
                "lr": lr,
                "lr_patience": lr_patience,
                "weight_decay": regularization,
                "use_scheduler": True,
            }
            model = None
            if warmup_period > 0:
                logging.info('running warmup...')
                model = train_mean_var_nn(
                    n_iter=warmup_period, train_var=False, frozen_var_value=frozen_var_value, do_plot_losses=False,
                    **common_params
                )
            logging.info('running main training run...')
            model = train_mean_var_nn(
                model=model, n_iter=n_iter, train_var=True, do_plot_losses=False,
                **common_params
            )
            if save_model:
                logging.info('saving model...')
                model.eval()
                self.io_helper.save_torch_model_statedict(model, filename)

        X_pred = misc_helpers.np_array_to_tensor(X_pred)
        X_pred = misc_helpers.object_to_cuda(X_pred)
        X_pred = misc_helpers.make_tensor_contiguous(X_pred)
        with torch.no_grad():
            # noinspection PyUnboundLocalVariable,PyCallingNonCallable
            y_pred, y_var = model(X_pred)
        y_pred, y_var = misc_helpers.tensors_to_np_arrays(y_pred, y_var)
        y_std = np.sqrt(y_var)
        y_quantiles = self.quantiles_gaussian(quantiles, y_pred, y_std)
        return y_pred, y_quantiles, y_std

    def native_gpytorch(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_pred: np.ndarray,
            quantiles,
            n_iter=100,
            val_frac=0.1,
            lr=1e-2,
            use_scheduler=True,
            lr_patience=30,
            lr_reduction_factor=0.5,
            show_progress=True,
            show_plots=True,
            do_plot_losses=True,
            skip_training=True,
            save_model=True,
    ):
        import torch
        import gpytorch
        from src_uq_methods_native.gp_regression_gpytorch import ExactGPModel, train_gpytorch
        from helpers import misc_helpers

        logging.info('preparing data..')
        X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_frac)
        X_train, y_train, X_val, y_val, X_pred = misc_helpers.np_arrays_to_tensors(
            X_train, y_train, X_val, y_val, X_pred
        )
        y_train, y_val = misc_helpers.make_ys_1d(y_train, y_val)

        logging.info('mapping data to device and making it contiguous...')
        X_train, y_train, X_val, y_val, X_pred = misc_helpers.objects_to_cuda(X_train, y_train, X_val, y_val, X_pred)
        X_train, y_train, X_val, y_val, X_pred = misc_helpers.make_tensors_contiguous(
            X_train, y_train, X_val, y_val, X_pred
        )

        n_samples = X_train.shape[0]
        common_postfix = f'n{n_samples}_it{n_iter}'
        common_prefix = 'native_gpytorch'
        filename_model = f'{common_prefix}_{common_postfix}.pth'
        filename_likelihood = f'{common_prefix}_likelihood_{common_postfix}.pth'
        if skip_training:
            logging.info('skipping training...')
            try:
                likelihood = self.io_helper.load_torch_model_statedict(gpytorch.likelihoods.GaussianLikelihood,
                                                                       filename_likelihood)
                model = self.io_helper.load_torch_model_statedict(ExactGPModel, filename_model,
                                                                  X_train=X_train, y_train=y_train,
                                                                  likelihood=likelihood)
                model, likelihood = misc_helpers.objects_to_cuda(model, likelihood)
            except FileNotFoundError:
                logging.warning(f'error: cannot load models {filename_model} and/or {filename_likelihood}')
                skip_training = False

        if not skip_training:
            logging.info('training from scratch...')
            model, likelihood = train_gpytorch(
                X_train,
                y_train,
                X_val,
                y_val,
                n_iter=n_iter,
                lr=lr,
                use_scheduler=use_scheduler,
                lr_patience=lr_patience,
                lr_reduction_factor=lr_reduction_factor,
                show_progress=show_progress,
                show_plots=show_plots,
                do_plot_losses=do_plot_losses,
            )
            if save_model:
                logging.info('saving model...')
                model.eval()
                likelihood.eval()
                self.io_helper.save_torch_model_statedict(model, filename_model)
                self.io_helper.save_torch_model_statedict(likelihood, filename_likelihood)

        # noinspection PyUnboundLocalVariable
        model.eval()
        # noinspection PyUnboundLocalVariable
        likelihood.eval()
        with torch.no_grad():  # todo: use gpytorch.settings.fast_pred_var()?
            f_preds = model(X_pred)
        y_preds = f_preds.mean
        y_std = f_preds.stddev
        y_preds, y_std = misc_helpers.tensors_to_np_arrays(y_preds, y_std)

        y_quantiles = self.quantiles_gaussian(quantiles, y_preds, y_std)
        return y_preds, y_quantiles, y_std

    @staticmethod
    def quantiles_gaussian(quantiles, y_pred: np.ndarray, y_std: np.ndarray):
        from scipy.stats import norm
        # todo: does this work for multi-dim outputs?
        return np.array([norm.ppf(quantiles, loc=mean, scale=std)
                         for mean, std in zip(y_pred, y_std)])

    @staticmethod
    def _clean_ys_for_metrics(*ys) -> Generator[np.ndarray | None, None, None]:
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


def check_method_kwargs_dict(class_, method_kwargs_dict):
    from inspect import signature
    wrong_kwargs = {}
    for method_name, method_kwargs in method_kwargs_dict.items():
        method = getattr(class_, method_name)
        method_params_names = set(signature(method).parameters)
        method_params_names.discard('self')
        kwargs_names = set(method_kwargs)
        if not kwargs_names.issubset(method_params_names):
            wrong_kwargs[method_name] = kwargs_names.difference(method_params_names)
    if wrong_kwargs:
        raise ValueError(f'Wrong method(s) kwargs: {wrong_kwargs}')
    logging.info('kwargs dict check successful')


def main():
    check_method_kwargs_dict(UQ_Comparison_Pipeline, METHODS_KWARGS)

    import torch
    torch.set_default_dtype(torch.float32)

    uq_comparer = UQ_Comparison_Pipeline(
        storage_path=STORAGE_PATH,
        methods_kwargs=METHODS_KWARGS,
        method_whitelist=METHOD_WHITELIST,
        posthoc_base_blacklist=POSTHOC_BASE_BLACKLIST,
        n_points_per_group=N_POINTS_PER_GROUP,
    )
    uq_comparer.compare_methods(
        QUANTILES,
        should_plot_data=PLOT_DATA,
        should_plot_uq_results=PLOT_UQ_RESULTS,
        should_plot_base_results=PLOT_BASE_RESULTS,
        should_show_plots=SHOW_PLOTS,
        should_save_plots=SAVE_PLOTS,
        should_save_results=SHOULD_SAVE_RESULTS,
        skip_base_model_copy=SKIP_BASE_MODEL_COPY,
    )


if __name__ == "__main__":
    main()
