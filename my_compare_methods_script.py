import numpy as np

from compare_methods import UQ_Comparer, print_metrics

from helpers import get_data, standardize, train_val_split, make_ys_1d, \
    np_arrays_to_tensors, make_tensors_contiguous, tensors_to_device, dfs_to_np_arrays
from io_helper import IO_Helper

METHOD_WHITELIST = [
    # "posthoc_conformal_prediction",
    "posthoc_laplace",
    # "native_quantile_regression",
    # "native_gpytorch",
    # "native_mvnn",
]
QUANTILES = [0.05, 0.25, 0.75, 0.95]  # todo: how to handle 0.5? ==> just use mean if needed

DATA_FILEPATH = './data.pkl'

N_POINTS_PER_GROUP = 100
PLOT_DATA = False
PLOT_RESULTS = True
SHOW_PLOTS = True
SAVE_PLOTS = True

TEST_BASE_MODEL_ONLY = False
TEST_RUN_ALL_BASE_MODELS = False
BASE_MODEL = 'NN'

PLOTS_PATH = "comparison_storage/plots"

METHODS_KWARGS = {
    "native_mvnn": {
        "n_iter": 300,
        "lr": 1e-4,
        "lr_patience": 30,
        "regularization": 0,  # 1e-2,
        "warmup_period": 50,
        "frozen_var_value": 0.1,
        'skip_training': False,
        'save_model': True,
    },
    "native_quantile_regression": {
        'skip_training': False,
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
        "n_iter": 300,
        'skip_training': True,
        'save_model': True,
    },
    "base_model": {  # todo: split
        "n_iter": 100,
        "verbose": 1,
        "show_progress_bar": True,
        "show_losses_plot": False,
        "save_losses_plot": True,
        "random_state": 42,
        "lr": 1e-2,
        "lr_reduction_factor": 0.5,
        "lr_patience": 30,
        "cv_n_iter": 5,
        "skip_training": True,
        "save_model": True,
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
        super().__init__(*args, **kwargs)
        if methods_kwargs is not None:
            self.methods_kwargs.update(methods_kwargs)
        self.io_helper = IO_Helper(storage_path)
        self.to_standardize = to_standardize
        self.n_points_per_group = n_points_per_group

    # todo: remove param?
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

    # todo: type hints!
    def compute_metrics(
            self, y_pred, y_quantiles, y_std, y_true, quantiles=None
    ) -> dict[str, float]:
        """

        :param y_pred: predicted y-values
        :param y_quantiles:
        :param y_std:
        :param y_true:
        :param quantiles:
        :return:
        """
        from metrics import rmse, smape_scaled, crps, nll_gaussian, mean_pinball_loss

        # todo: sharpness? calibration? PIT? coverage?
        # todo: skill score (but what to use as benchmark)?

        def clean_y(y):
            if y is None:
                return y
            return np.array(y).squeeze()

        y_pred, y_quantiles, y_std, y_true = map(clean_y, (y_pred, y_quantiles, y_std, y_true))
        metrics = {
            "rmse": rmse(y_true, y_pred),
            "smape_scaled": smape_scaled(y_true, y_pred),
            "crps": crps(y_true, y_pred, y_std),
            "nll_gaussian": nll_gaussian(y_true, y_pred, y_std),
            "mean_pinball": mean_pinball_loss(y_pred, y_quantiles, quantiles),
        }
        metrics = {
            key: (float(value) if value is not None else value)
            for key, value in metrics.items()
        }
        return metrics

    def train_base_model(self, *args, **kwargs):
        # todo: more flexibility in choosing (multiple) base models
        if TEST_RUN_ALL_BASE_MODELS:
            model = self.my_train_base_model_rf(*args, **kwargs)
            model = self.my_train_base_model_nn(*args, **kwargs)
        else:
            if BASE_MODEL == 'RF':
                model = self.my_train_base_model_rf(*args, **kwargs)
            elif BASE_MODEL == 'NN':
                model = self.my_train_base_model_nn(*args, **kwargs)
            else:
                raise ValueError(f'Unknown base model type: {BASE_MODEL}')
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
            save_model=True,
            verbose=True,
            **kwargs,
    ):
        """

        :param X_train:
        :param y_train:
        :param model_params_choices:
        :param model_init_params:
        :param skip_training:
        :param n_jobs:
        :param cv_n_iter:
        :param save_model:
        :param verbose:
        :param kwargs: unused kwargs (for other base model)
        :return:
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
        from scipy.stats import randint

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
        print("done.")
        if save_model:
            print('saving model...')
            self.io_helper.save_model(model, filename_base_model)
        return model

    def my_train_base_model_nn(
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
            **kwargs,
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
        :param kwargs: unused kwargs (for other base model)
        :return:
        """
        from nn_estimator import NN_Estimator

        X_train, y_train = np_arrays_to_tensors(X_train, y_train)
        X_train, y_train = tensors_to_device(X_train, y_train)

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
            X_uq,
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
        :param X_uq: 
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
            X_uq,
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
            X_uq: np.ndarray,
            quantiles,
            base_model: "NN_Estimator",
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
        from helpers import get_train_loader, tensor_to_np_array
        import torch
        from torch import nn

        torch.set_default_dtype(torch.float32)
        torch.manual_seed(random_seed)

        def la_instantiator(base_model: nn.Module):
            return Laplace(base_model, "regression")

        n_training_points = X_train.shape[0]
        if model_filename is None:
            model_filename = f"laplace_{n_training_points}_{n_iter}.pth"
        if skip_training:
            print("skipping base model training...")
            try:
                # noinspection PyTypeChecker
                la = self.io_helper.load_laplace_model_statedict(
                    base_model,
                    la_instantiator,
                    laplace_model_filename=model_filename,
                )
            except FileNotFoundError:
                print(f"error. model {model_filename} not found, so training cannot be skipped. training from scratch.")
                skip_training = False

        if not skip_training:
            X_uq, X_train, y_train = np_arrays_to_tensors(X_uq, X_train, y_train)
            X_uq, X_train, y_train = tensors_to_device(X_uq, X_train, y_train)
            train_loader = get_train_loader(X_train, y_train, batch_size)
            la = la_instantiator(base_model.get_nn())
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

        f_mu, f_var = la(X_uq)
        f_mu = tensor_to_np_array(f_mu.squeeze())
        f_sigma = tensor_to_np_array(f_var.squeeze().sqrt())
        pred_std = np.sqrt(f_sigma ** 2 + la.sigma_noise.item() ** 2)

        y_pred, y_std = f_mu, pred_std
        y_quantiles = self.quantiles_gaussian(quantiles, y_pred, y_std)
        return y_pred, y_quantiles, y_std

    def native_quantile_regression(self, X_train: np.ndarray, y_train: np.ndarray, X_uq: np.ndarray, quantiles,
                                   verbose=True,
            skip_training=True,
            save_model=True,):
        from quantile_regression import estimate_quantiles as estimate_quantiles_qr
        y_pred, y_quantiles = estimate_quantiles_qr(
            X_train, y_train, X_uq, alpha=quantiles
        )
        y_std = self.stds_from_quantiles(y_quantiles)
        return y_pred, y_quantiles, y_std

    @staticmethod
    def native_mvnn(X_train: np.ndarray, y_train: np.ndarray, X_uq: np.ndarray, quantiles,
            skip_training=True,
            save_model=True,):
        from mean_var_nn import run_mean_var_nn
        return run_mean_var_nn(
            X_train,
            y_train,
            X_uq,
            quantiles,
            skip_training=True,
            save_model=True,
        )

    def native_gpytorch(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_uq: np.ndarray,
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

        print('preparing data..')
        X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_frac)
        X_train, y_train, X_val, y_val, X_uq = np_arrays_to_tensors(X_train, y_train, X_val, y_val, X_uq)
        y_train, y_val = make_ys_1d(y_train, y_val)

        print('making data contiguous and mapping to device...')
        X_train, y_train, X_val, y_val, X_uq = make_tensors_contiguous(X_train, y_train, X_val, y_val, X_uq)
        X_train, y_train, X_val, y_val, X_uq = tensors_to_device(X_train, y_train, X_val, y_val, X_uq)

        common_prefix, common_postfix = f'{model_name}', f'{self.n_points_per_group}_{n_epochs}'
        model_name, model_likelihood_name = f'{common_prefix}_{common_postfix}.pth', f'{common_prefix}_likelihood_{common_postfix}.pth'
        if skip_training:
            print('skipping training...')
            try:
                likelihood = self.io_helper.load_torch_model_statedict(gpytorch.likelihoods.GaussianLikelihood,
                                                                       model_likelihood_name)
                model = self.io_helper.load_torch_model_statedict(ExactGPModel, model_name,
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
                n_epochs,
                lr=lr,
                show_progress=True,
                show_plots=True,
                do_plot_losses=True,
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

        with torch.no_grad():
            f_preds = model(X_uq)
        y_preds = f_preds.mean
        y_std = f_preds.stddev
        y_quantiles = self.quantiles_gaussian(quantiles, y_preds, y_std)
        return y_preds, y_quantiles, y_std

    @staticmethod
    def quantiles_gaussian(quantiles, y_pred, y_std):
        from scipy.stats import norm
        # todo: does this work for multi-dim outputs?
        return np.array([norm.ppf(quantiles, loc=mean, scale=std)
                         for mean, std in zip(y_pred, y_std)])

    def _standardize_or_to_array(self, variable, *dfs):
        if variable in self.to_standardize:
            return standardize(*dfs, return_scaler=False)
        return dfs_to_np_arrays(dfs)

    def plot_post_training_perf(self, base_model, X_train, y_train, do_save_figure=False, filename='base_model'):
        from matplotlib import pyplot as plt

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
            plot_path='plots',
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
            plt.show()
        else:
            plt.close(fig)


def test_base_model():
    print('running base model test')
    import torch
    torch.set_default_dtype(torch.float32)

    print('loading data...')
    uq_comparer = My_UQ_Comparer(
        methods_kwargs=METHODS_KWARGS,
        method_whitelist=METHOD_WHITELIST,
        to_standardize=TO_STANDARDIZE,
        n_points_per_group=N_POINTS_PER_GROUP,
    )
    X_train, X_test, y_train, y_test, X, y = uq_comparer.get_data()
    X_uq = np.row_stack((X_train, X_test))

    print("training base model...")
    base_model_kwargs = uq_comparer.methods_kwargs['base_model']
    base_model = uq_comparer.train_base_model(X_train, y_train, **base_model_kwargs)

    print('predicting...')
    y_preds = base_model.predict(X_uq)

    print('plotting...')
    uq_comparer.plot_base_model_test_result(
        X_train,
        X_test,
        y_train,
        y_test,
        y_preds,
        plot_name='base_model_test',
        show_plots=SHOW_PLOTS,
        save_plot=SAVE_PLOTS,
        plot_path=PLOTS_PATH,
    )
    print('done.')


def main():
    import torch
    torch.set_default_dtype(torch.float32)

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


if __name__ == "__main__":
    if TEST_BASE_MODEL_ONLY:
        test_base_model()
    else:
        main()
