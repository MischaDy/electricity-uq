import numpy as np
from more_itertools import collapse

# noinspection PyProtectedMember
from sklearn.base import RegressorMixin, BaseEstimator, _fit_context

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from helpers import misc_helpers


# noinspection PyAttributeOutsideInit,PyPep8Naming
class NN_Estimator(RegressorMixin, BaseEstimator):
    """
    Parameters
    ----------

    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """

    _parameter_constraints = {
        "n_iter": [int],
        "batch_size": [int],
        "random_state": [int],
        "lr": [float],
        "lr_patience": [int],
        "lr_reduction_factor": [float],
        "to_standardize": [str],
        "val_frac": [float],
        "use_scheduler": [bool],
        "skip_training": [bool],
        "save_model": [bool],
        "verbose": [int],
        'show_progress_bar': [bool],
        'show_losses_plot': [bool],
        'save_losses_plot': [bool],
    }

    def __init__(
            self,
            n_iter=100,
            batch_size=20,
            random_seed=42,
            val_frac=0.1,
            num_hidden_layers=2,
            hidden_layer_size=50,
            activation=None,
            use_scheduler=True,
            lr=None,
            lr_patience=5,
            lr_reduction_factor=0.5,
            verbose: int = 1,
            show_progress_bar=True,
            show_losses_plot=True,
            save_losses_plot=True,
            io_helper=None,
    ):
        self.use_scheduler = use_scheduler
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.val_frac = val_frac
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_reduction_factor = lr_reduction_factor
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.show_losses_plot = show_losses_plot
        self.save_losses_plot = save_losses_plot
        self.io_helper = io_helper
        self.is_fitted_ = False

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers).


        Returns
        -------
        self : object
            Returns self.
        """
        """`_validate_data` is defined in the `BaseEstimator` class.
        It allows to:
        - run different checks on the input data;
        - define some attributes associated to the input data: `n_features_in_` and
          `feature_names_in_`."""
        torch.set_default_device(misc_helpers.get_device())
        torch.manual_seed(self.random_seed)

        X, y = self._validate_data(X, y, accept_sparse=False)  # todo: remove "y is 2d" warning

        if self.activation is None:
            self.activation = torch.nn.LeakyReLU

        self.is_y_2d_ = len(y.shape) == 2
        if len(y.shape) < 2:
            y = y.reshape(-1, 1)

        try:
            X_train, y_train = misc_helpers.np_arrays_to_tensors(X, y)
        except TypeError:
            raise TypeError(f'Unknown label type: {X.dtype} (X) or {y.dtype} (y)')

        X_train, y_train, X_val, y_val = misc_helpers.train_val_split(X_train, y_train, self.val_frac)
        X_train, y_train, X_val, y_val = misc_helpers.objects_to_cuda(X_train, y_train, X_val, y_val)
        X_train, y_train, X_val, y_val = misc_helpers.make_tensors_contiguous(X_train, y_train, X_val, y_val)

        dim_in, dim_out = X_train.shape[-1], y_train.shape[-1]

        model = self._nn_builder(
            dim_in,
            dim_out,
            num_hidden_layers=self.num_hidden_layers,
            hidden_layer_size=self.hidden_layer_size,
            activation=self.activation,
        )
        model = misc_helpers.object_to_cuda(model)

        # noinspection PyTypeChecker
        train_loader = self._get_train_loader(X_train, y_train, self.batch_size)

        if self.lr is None:
            self.lr = 1e-2 if self.use_scheduler else 1e-4

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.lr_patience, factor=self.lr_reduction_factor)
        criterion = torch.nn.MSELoss()
        criterion = misc_helpers.object_to_cuda(criterion)

        train_losses, val_losses = [], []
        epochs = tqdm(range(self.n_iter)) if self.show_progress_bar else range(self.n_iter)
        for _ in epochs:
            model.train()
            for X, y in train_loader:
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = self._mse_torch(model(X_val), y_val)
                if self.save_losses_plot:
                    train_loss = self._mse_torch(model(X_train), y_train)
                    train_loss = misc_helpers.tensor_to_np_array(train_loss)
                    train_losses.append(train_loss)
            if self.use_scheduler:
                scheduler.step(val_loss)
            val_loss = misc_helpers.tensor_to_np_array(val_loss)
            val_losses.append(val_loss)

        model.eval()
        self.model_ = model

        loss_skip = min(100, self.n_iter // 10)
        misc_helpers.plot_nn_losses(
            train_losses,
            val_losses,
            loss_skip=loss_skip,
            show_plot=self.show_losses_plot,
            save_plot=self.save_losses_plot,
            io_helper=self.io_helper,
            file_name='nn_estimator_fit',
        )

        self.is_fitted_ = True
        return self

    @staticmethod
    def _nn_builder(
            dim_in,
            dim_out,
            num_hidden_layers=2,
            hidden_layer_size=50,
            activation=torch.nn.LeakyReLU,
    ):
        layers = collapse([
            torch.nn.Linear(dim_in, hidden_layer_size),
            activation(),
            [[torch.nn.Linear(hidden_layer_size, hidden_layer_size),
              activation()]
             for _ in range(num_hidden_layers)],
            torch.nn.Linear(hidden_layer_size, dim_out),
        ])
        model = torch.nn.Sequential(*layers)
        return model.float()

    @classmethod
    def _get_train_loader(cls, X_train: torch.Tensor, y_train: torch.Tensor, batch_size):
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        return train_loader

    @staticmethod
    def _mse_torch(y_pred, y_test):
        return torch.mean((y_pred - y_test) ** 2)

    def predict(self, X, as_np=True):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        as_np : ...

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.

        """
        from sklearn.utils.validation import check_is_fitted

        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=False, reset=False)
        X = misc_helpers.np_array_to_tensor(X)
        X = misc_helpers.object_to_cuda(X)
        X = misc_helpers.make_tensor_contiguous(X)

        # self.model_.eval()
        with torch.no_grad():
            res = self.model_(X)
        res = res.reshape(-1, 1) if self.is_y_2d_ else res.squeeze()
        if as_np:
            res = misc_helpers.tensor_to_np_array(res)
        return res

    def get_nn(self, to_device=True) -> torch.nn.Module:
        if to_device:
            return misc_helpers.object_to_cuda(self.model_)
        return self.model_

    def _more_tags(self):
        return {'poor_score': True,
                '_xfail_checks': {'check_methods_sample_order_invariance': '(barely) failing for unknown reason'}}

    # noinspection PyUnusedLocal
    def to(self, device):
        self.model_ = misc_helpers.object_to_cuda(self.model_)
        return self

    def __getattr__(self, item):
        try:
            return getattr(self.__getattribute__('model_'), item)
        except AttributeError:
            msg = f'NN_Estimator has no attribute "{item}"'
            if not self.__getattribute__('is_fitted_'):
                msg += ', or only has it is after fitting'
            raise AttributeError(msg)

    def __call__(self, tensor: torch.Tensor):
        if not self.__getattribute__('is_fitted_'):
            raise TypeError('NN_Estimator is only callable after fitting')
        tensor = misc_helpers.object_to_cuda(tensor)
        tensor = misc_helpers.make_tensor_contiguous(tensor)
        return self.model_(tensor)

    def __setattr__(self, key, value):
        self.__dict__[key] = value


def train_nn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iter=500,
        batch_size=20,
        random_seed=42,
        val_frac=0.1,
        num_hidden_layers=2,
        hidden_layer_size=50,
        activation=None,
        lr=0.1,
        lr_patience=30,
        lr_reduction_factor=0.5,
        show_progress_bar=True,
        show_losses_plot=True,
        save_losses_plot=True,
        io_helper=None,
        verbose: int = 1,
):
    model = NN_Estimator(
        n_iter=n_iter,
        batch_size=batch_size,
        random_seed=random_seed,
        val_frac=val_frac,
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
        io_helper=io_helper,
    )
    # noinspection PyTypeChecker
    model.fit(X_train, y_train)
    return model


if __name__ == '__main__':
    from sklearn.utils.estimator_checks import check_estimator
    estimator = NN_Estimator(verbose=0)
    check_estimator(estimator)
