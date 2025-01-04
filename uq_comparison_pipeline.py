import os
import logging

filename = os.path.split(__file__)[-1]
logging.info(f'reading file {filename}...')

from typing import Any, Generator, Union, TYPE_CHECKING

from uq_comparison_pipeline_abc import UQ_Comparison_Pipeline_ABC
from helpers import misc_helpers
import settings

if TYPE_CHECKING:
    from src_base_models.nn_estimator import NN_Estimator
    import numpy as np
    from helpers.model_wrapper import ModelWrapper


# noinspection PyPep8Naming
class UQ_Comparison_Pipeline(UQ_Comparison_Pipeline_ABC):
    posthoc_base_blacklist = {
        'posthoc_laplace_approximation': {
            'base_model_linreg',
            'base_model_rf',
        },
    }

    def __init__(
            self,
            filename_parts,
            data_path,
            *,
            storage_path="comparison_storage",
            methods_kwargs: dict[str, dict[str, Any]] = None,
            n_points_per_group=None,
            method_whitelist=None,
            do_standardize_data=True,
    ):
        """
        :param methods_kwargs: dict of (method_name, method_kwargs) pairs
        :param storage_path:
        :param n_points_per_group: both training size and test size
        :param do_standardize_data: True if both X and y should be standardized, False if neither.
        """
        super().__init__(
            storage_path=storage_path,
            data_path=data_path,
            methods_kwargs=methods_kwargs,
            filename_parts=filename_parts,
            method_whitelist=method_whitelist,
            do_standardize_data=do_standardize_data,
        )
        self.n_points_per_group = n_points_per_group
        self.train_years = settings.TRAIN_YEARS
        self.val_years = settings.VAL_YEARS
        self.test_years = settings.TEST_YEARS

    def get_data(self):
        """
        load and prepare data

        :return:
        A tuple (X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y).
        If self.standardize_data=False, y_scaler is None. All variables except for the scaler are 2D np arrays.
        """
        return misc_helpers.get_data(
            filepath=self.data_path,
            train_years=self.train_years,
            val_years=self.val_years,
            test_years=self.test_years,
            n_points_per_group=self.n_points_per_group,
            do_standardize_data=self.do_standardize_data,
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
    def compute_metrics_uq(cls, y_pred, y_quantiles, y_std, y_true, quantiles) -> dict[str, float]:
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
            X_train: 'np.ndarray',
            y_train: 'np.ndarray',
            X_val: 'np.ndarray',
            y_val: 'np.ndarray',
            n_jobs=-1,
            skip_training=True,
            save_model=True,
    ) -> 'ModelWrapper':
        from src_base_models.linear_regression import train_linreg
        from helpers.model_wrapper import ModelWrapper

        method_name = 'base_model_linreg'
        if skip_training:
            model = self.try_skipping_training(method_name)
            if model is None:
                skip_training = False
        if not skip_training:
            model = train_linreg(X_train, y_train, X_val, y_val, n_jobs=n_jobs)
            if save_model:
                self.save_model(model, method_name=method_name)
        # noinspection PyUnboundLocalVariable
        y_pred_temp = model.predict(X_train[:1])
        model_wrapped = ModelWrapper(model, output_dim=len(y_pred_temp.shape))
        return model_wrapped

    def base_model_rf(
            self,
            X_train: 'np.ndarray',
            y_train: 'np.ndarray',
            X_val: 'np.ndarray',
            y_val: 'np.ndarray',
            cv_n_iter=100,
            cv_n_splits=10,
            model_param_distributions=None,
            n_jobs=-1,
            random_seed=42,
            skip_training=True,
            save_model=True,
            verbose=1,
    ) -> 'ModelWrapper':
        from src_base_models.random_forest import train_random_forest
        from helpers.model_wrapper import ModelWrapper
        method_name = 'base_model_rf'
        if skip_training:
            model = self.try_skipping_training(method_name)
            if model is None:
                skip_training = False
        if not skip_training:
            assert all(item is not None for item in [X_train, y_train, model_param_distributions])
            model = train_random_forest(
                X_train,
                y_train,
                X_val,
                y_val,
                cv_n_iter=cv_n_iter,
                cv_n_splits=cv_n_splits,
                model_param_distributions=model_param_distributions,
                random_seed=random_seed,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            if save_model:
                self.save_model(model, method_name=method_name)
        # noinspection PyUnboundLocalVariable
        y_pred_temp = model.predict(X_train[:1])
        model_wrapped = ModelWrapper(model, output_dim=len(y_pred_temp.shape))
        return model_wrapped

    def base_model_nn(
            self,
            X_train: 'np.ndarray',
            y_train: 'np.ndarray',
            X_val: 'np.ndarray',
            y_val: 'np.ndarray',
            n_iter=500,
            batch_size=20,
            random_seed=42,
            num_hidden_layers=2,
            hidden_layer_size=50,
            activation=None,
            lr=0.1,
            lr_patience=30,
            lr_reduction_factor=0.5,
            show_progress_bar=True,
            show_losses_plot=True,
            save_losses_plot=True,
            skip_training=True,
            save_model=True,
            verbose: int = 1,
    ) -> 'NN_Estimator':
        """

        :param y_val:
        :param X_val:
        :param activation:
        :param hidden_layer_size:
        :param num_hidden_layers:
        :param show_losses_plot:
        :param save_losses_plot:
        :param show_progress_bar:
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
        from src_base_models.nn_estimator import train_nn
        import torch
        torch.set_default_dtype(torch.float32)

        if not torch.cuda.is_available():
            logging.warning('cuda not available! using CPU')

        method_name = 'base_model_nn'
        if skip_training:
            model = self.try_skipping_training(method_name)
            if model is not None:
                return model

        model = train_nn(
            X_train,
            y_train,
            X_val,
            y_val,
            n_iter=n_iter,
            batch_size=batch_size,
            random_seed=random_seed,
            num_hidden_layers=num_hidden_layers,
            hidden_layer_size=hidden_layer_size,
            activation=activation,
            lr=lr,
            lr_patience=lr_patience,
            lr_reduction_factor=lr_reduction_factor,
            verbose=verbose,
            show_progress_bar=show_progress_bar,
            show_losses_plot=show_losses_plot,
            save_losses_plot=save_losses_plot,
            io_helper=self.io_helper,
        )
        with torch.no_grad():
            y_pred_temp = model.predict(X_train[:1], as_np=False)
        model.set_output_dim(len(y_pred_temp.shape), orig=True)
        if save_model:
            self.save_model(model, method_name=method_name)
        # noinspection PyTypeChecker
        model.set_params(verbose=False)
        return model

    def posthoc_conformal_prediction(
            self,
            X_train: 'np.ndarray',
            y_train: 'np.ndarray',
            X_val: 'np.ndarray',
            y_val: 'np.ndarray',
            X_pred: 'np.ndarray',
            quantiles: list,
            base_model: Union['ModelWrapper', 'NN_Estimator'],
            n_estimators=10,
            bootstrap_n_blocks=10,
            bootstrap_overlapping_blocks=False,
            random_seed=42,
            verbose=1,
            skip_training=True,
            save_model=True,
    ):
        """

        :param y_val:
        :param X_val:
        :param save_model:
        :param skip_training:
        :param verbose:
        :param X_train:
        :param y_train:
        :param X_pred:
        :param quantiles:
        :param base_model:
        :param random_seed:
        :param n_estimators: number of model clones to train for ensemble
        :param bootstrap_n_blocks:
        :param bootstrap_overlapping_blocks:
        :return:
        """
        from src_uq_methods_posthoc.conformal_prediction import (
            train_conformal_prediction,
            predict_with_conformal_prediction,
        )
        method_name = 'posthoc_conformal_prediction'
        if skip_training:
            model = self.try_skipping_training(method_name)
            if model is None:
                skip_training = False
        if not skip_training:
            base_model.set_output_dim(1)
            model = train_conformal_prediction(
                X_train,
                y_train,
                X_val,
                y_val,
                base_model,
                n_estimators=n_estimators,
                bootstrap_n_blocks=bootstrap_n_blocks,
                bootstrap_overlapping_blocks=bootstrap_overlapping_blocks,
                random_seed=random_seed,
                verbose=verbose,
            )
            if save_model:
                self.save_model(model, method_name=method_name)
        # noinspection PyUnboundLocalVariable
        y_pred, y_quantiles, y_std = predict_with_conformal_prediction(model, X_pred, quantiles)
        base_model.reset_output_dim()
        return y_pred, y_quantiles, y_std

    def posthoc_laplace_approximation(
            self,
            X_train: 'np.ndarray',
            y_train: 'np.ndarray',
            X_val: 'np.ndarray',
            y_val: 'np.ndarray',
            X_pred: 'np.ndarray',
            quantiles: list,
            base_model: 'NN_Estimator',
            n_iter=100,
            batch_size=20,
            random_seed=42,
            skip_training=True,
            save_model=True,
            verbose=True,
            show_progress=True,
    ):
        from src_uq_methods_posthoc.laplace_approximation import (
            la_instantiator, train_laplace_approximation, predict_with_laplace_approximation
        )
        method_name = 'posthoc_laplace_approximation'
        base_model_nn = base_model.get_nn(to_device=True)
        if skip_training:
            logging.info(f'skipping model training in method {method_name}')
            try:
                # noinspection PyTypeChecker
                model = self.io_helper.load_laplace_model_statedict(
                    base_model_nn,
                    la_instantiator,
                    method_name=method_name,
                )
            except FileNotFoundError as error:
                logging.warning(f"trained model '{error.filename}' not found. training from scratch.")
                skip_training = False

        if not skip_training:
            model = train_laplace_approximation(
                X_train,
                y_train,
                X_val,
                y_val,
                base_model_nn,
                n_iter=n_iter,
                batch_size=batch_size,
                random_seed=random_seed,
                verbose=verbose,
                show_progress=show_progress,
            )
            if save_model:
                logging.info('saving model...')
                self.io_helper.save_laplace_model_statedict(model, method_name=method_name)
        # noinspection PyUnboundLocalVariable
        y_pred, y_quantiles, y_std = predict_with_laplace_approximation(model, X_pred, quantiles)
        return y_pred, y_quantiles, y_std

    def native_quantile_regression_nn(
            self,
            X_train: 'np.ndarray',
            y_train: 'np.ndarray',
            X_val: 'np.ndarray',
            y_val: 'np.ndarray',
            X_pred: 'np.ndarray',
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
            show_losses_plot=True,
            save_losses_plot=True,
            skip_training=True,
            save_model=True,
    ):
        import torch
        torch.set_default_dtype(torch.float32)

        if not torch.cuda.is_available():
            logging.warning('cuda not available! using CPU')

        from src_uq_methods_native.quantile_regression_nn import (
            QR_NN,
            train_qr_nn,
            predict_with_qr_nn,
        )
        torch.manual_seed(random_seed)
        method_name = 'native_quantile_regression_nn'
        if skip_training:
            logging.info('skipping training...')
            if activation is None:
                activation = torch.nn.LeakyReLU
            try:
                model = self.io_helper.load_torch_model_statedict(
                    QR_NN,
                    method_name=method_name,
                    model_kwargs={
                        'dim_in': X_train.shape[1],
                        'quantiles': quantiles,
                        'num_hidden_layers': num_hidden_layers,
                        'hidden_layer_size': hidden_layer_size,
                        'activation': activation,
                    },
                )
                model = misc_helpers.object_to_cuda(model)
            except FileNotFoundError as error:
                logging.warning(f"trained model '{error.filename}' not found.")
                skip_training = False

        if not skip_training:
            logging.info('training from scratch...')
            model = train_qr_nn(
                X_train,
                y_train,
                X_val,
                y_val,
                quantiles,
                n_iter=n_iter,
                lr=lr,
                use_scheduler=use_scheduler,
                lr_patience=lr_patience,
                weight_decay=regularization,
                show_losses_plot=show_losses_plot,
                save_losses_plot=save_losses_plot,
                io_helper=self.io_helper,
                show_progress=show_progress,
            )
            if save_model:
                logging.info('saving model...')
                model.eval()
                self.io_helper.save_torch_model_statedict(model, method_name=method_name)
        # noinspection PyUnboundLocalVariable
        y_pred, y_quantiles, y_std = predict_with_qr_nn(model, X_pred)
        return y_pred, y_quantiles, y_std

    def native_mvnn(
            self,
            X_train: 'np.ndarray',
            y_train: 'np.ndarray',
            X_val: 'np.ndarray',
            y_val: 'np.ndarray',
            X_pred: 'np.ndarray',
            quantiles: list,
            n_iter=300,
            num_hidden_layers=2,
            hidden_layer_size=50,
            activation=None,
            lr=1e-4,
            lr_patience=30,
            regularization=0,  # 1e-2,
            warmup_period=50,
            frozen_var_value=0.1,
            show_losses_plot=True,
            save_losses_plot=True,
            show_progress=True,
            skip_training=True,
            save_model=True,
    ):
        import torch
        from src_uq_methods_native.mean_var_nn import (
            MeanVarNN,
            train_mean_var_nn,
            predict_with_mvnn,
        )
        torch.set_default_dtype(torch.float32)

        if not torch.cuda.is_available():
            logging.warning('cuda not available! using CPU')

        method_name = 'native_mvnn'
        if skip_training:
            logging.info('skipping training...')
            if activation is None:
                activation = torch.nn.LeakyReLU
            try:
                model = self.io_helper.load_torch_model_statedict(
                    MeanVarNN,
                    method_name=method_name,
                    model_kwargs={
                        'dim_in': X_train.shape[1],
                        'num_hidden_layers': num_hidden_layers,
                        'hidden_layer_size': hidden_layer_size,
                        'activation': activation,
                    }
                )
                model = misc_helpers.object_to_cuda(model)
            except FileNotFoundError as error:
                logging.warning(f"trained model '{error.filename}' not found.")
                skip_training = False
        if not skip_training:
            logging.info('training from scratch...')
            common_params = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "lr": lr,
                "lr_patience": lr_patience,
                "weight_decay": regularization,
                'show_progress': show_progress,
                "use_scheduler": True,
            }
            model = None
            if warmup_period > 0:
                logging.info('running warmup...')
                model = train_mean_var_nn(
                    n_iter=warmup_period,
                    train_var=False,
                    frozen_var_value=frozen_var_value,
                    show_losses_plot=False,  # never show for warmup
                    save_losses_plot=False,  # never save for warmup
                    **common_params
                )
            logging.info('running main training run...')
            model = train_mean_var_nn(
                model=model,
                n_iter=n_iter,
                train_var=True,
                show_losses_plot=show_losses_plot,
                save_losses_plot=save_losses_plot,
                io_helper=self.io_helper,
                **common_params
            )
            if save_model:
                logging.info('saving model...')
                model.eval()
                self.io_helper.save_torch_model_statedict(model, method_name=method_name)
        # noinspection PyUnboundLocalVariable
        y_pred, y_quantiles, y_std = predict_with_mvnn(model, X_pred, quantiles)
        return y_pred, y_quantiles, y_std

    def native_gpytorch(
            self,
            X_train: 'np.ndarray',
            y_train: 'np.ndarray',
            X_val: 'np.ndarray',
            y_val: 'np.ndarray',
            X_pred: 'np.ndarray',
            quantiles: list,
            n_iter=100,
            lr=1e-2,
            use_scheduler=True,
            lr_patience=30,
            lr_reduction_factor=0.5,
            show_progress=True,
            show_plots=True,
            show_losses_plot=True,
            save_losses_plot=True,
            n_inducing_points=500,
            skip_training=True,
            save_model=True,
    ):
        import gpytorch
        import torch
        from src_uq_methods_native.gp_regression_gpytorch import (
            ApproximateGP,
            train_gpytorch,
            prepare_data,
            predict_with_gpytorch,
        )
        torch.set_default_dtype(torch.float32)

        if not torch.cuda.is_available():
            logging.warning('cuda not available! using CPU')

        X_train, y_train, X_val, y_val, X_pred = prepare_data(X_train, y_train, X_val, y_val, X_pred)
        method_name = 'native_gpytorch'
        infix = 'likelihood'
        if skip_training:
            logging.info(f'skipping training in method {method_name}...')
            try:
                likelihood = self.io_helper.load_torch_model_statedict(
                    gpytorch.likelihoods.GaussianLikelihood,
                    method_name=method_name,
                    infix=infix,
                )
                inducing_points = X_train[:n_inducing_points, :]
                model = self.io_helper.load_torch_model_statedict(
                    ApproximateGP,
                    method_name=method_name,
                    model_kwargs={'inducing_points': inducing_points},
                )
                model, likelihood = misc_helpers.objects_to_cuda(model, likelihood)
            except FileNotFoundError as error:
                logging.warning(f"trained model '{error.filename}' not found.")
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
                show_losses_plot=show_losses_plot,
                save_losses_plot=save_losses_plot,
                io_helper=self.io_helper,
                n_inducing_points=n_inducing_points,
            )
            if save_model:
                logging.info('saving model...')
                model.eval()
                likelihood.eval()
                self.io_helper.save_torch_model_statedict(model, method_name=method_name)
                self.io_helper.save_torch_model_statedict(likelihood, method_name=method_name, infix=infix)
        # noinspection PyUnboundLocalVariable
        y_pred, y_quantiles, y_std = predict_with_gpytorch(model, likelihood, X_pred, quantiles)
        return y_pred, y_quantiles, y_std

    @staticmethod
    def _clean_ys_for_metrics(*ys) -> Generator[Union['np.ndarray', None], None, None]:
        import numpy as np
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

    def try_skipping_training(self, method_name):
        try:
            logging.info(f'skipping model training in method {method_name}')
            model = self.io_helper.load_model(method_name=method_name)
            return model
        except FileNotFoundError as error:
            logging.warning(f"trained model '{error.filename}' not found. training from scratch.")
        return None

    def save_model(self, model, method_name):
        logging.info(f'saving model in method {method_name}...')
        self.io_helper.save_model(model, method_name=method_name)


def update_training_flags():
    logging.info('updating training flags...')
    assert not (settings.DO_TRAIN_ALL and settings.SKIP_TRAINING_ALL)

    for _, method_kwargs in settings.METHODS_KWARGS.items():
        if settings.DO_TRAIN_ALL:
            method_kwargs['skip_training'] = False
        elif settings.SKIP_TRAINING_ALL:
            method_kwargs['skip_training'] = True


def update_run_size_setup():
    assert not (settings.DO_BIG_RUN and settings.DO_SMALL_RUN)

    if settings.DO_BIG_RUN:
        settings.DATA_FILEPATH = 'data/data_2015_2018.pkl'  # 'data/data_1600.pkl'
        settings.N_POINTS_PER_GROUP = None
        settings.TRAIN_YEARS = (2016, 2017)
        settings.VAL_YEARS = (2017, 2018)
        settings.TEST_YEARS = (2018, 2019)
    elif settings.DO_SMALL_RUN:
        settings.DATA_FILEPATH = 'data/data_1600.pkl'
        settings.N_POINTS_PER_GROUP = 800
        settings.TRAIN_YEARS = None
        settings.VAL_YEARS = None
        settings.TEST_YEARS = None


def check_method_kwargs_dict(class_, method_kwargs_dict):
    logging.info('checking kwargs dict...')
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


def main():
    logging.basicConfig(level=settings.LOGGING_LEVEL, force=True)

    logging.info('running main pipeline...')
    logging.info('running preliminary checks/setup...')
    check_method_kwargs_dict(UQ_Comparison_Pipeline, settings.METHODS_KWARGS)
    update_run_size_setup()
    update_training_flags()
    # todo: check filename parts dict!

    uq_comparer = UQ_Comparison_Pipeline(
        filename_parts=settings.FILENAME_PARTS,
        data_path=settings.DATA_FILEPATH,
        storage_path=settings.STORAGE_PATH,
        methods_kwargs=settings.METHODS_KWARGS,
        method_whitelist=settings.METHOD_WHITELIST,
        n_points_per_group=settings.N_POINTS_PER_GROUP,
    )
    uq_comparer.compare_methods(
        settings.QUANTILES,
        should_plot_data=settings.PLOT_DATA,
        should_plot_uq_results=settings.PLOT_UQ_RESULTS,
        should_plot_base_results=settings.PLOT_BASE_RESULTS,
        should_show_plots=settings.SHOW_PLOTS,
        should_save_plots=settings.SAVE_PLOTS,
        should_save_results=settings.SHOULD_SAVE_RESULTS,
        skip_base_model_copy=settings.SKIP_BASE_MODEL_COPY,
        use_filesave_prefix=settings.USE_FILESAVE_PREFIX,
    )


if __name__ == "__main__":
    main()
