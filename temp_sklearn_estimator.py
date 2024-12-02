import numpy as np
import torch
from matplotlib import pyplot as plt
from more_itertools import collapse
from sklearn.base import RegressorMixin, BaseEstimator, _fit_context
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from helpers import IO_Helper


# noinspection PyAttributeOutsideInit
class MyEstimator(RegressorMixin, BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstration of how to pass and store parameters.

    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> class MyEstimator(RegressorMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         self.is_fitted_ = True
    ...         return self
    ...     def predict(self, X):
    ...         return np.full(shape=X.shape[0], fill_value=self.param)
    >>> estimator = MyEstimator(param=0)
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([-1, 0, 1])
    >>> estimator.fit(X, y).predict(X)
    array([0, 0, 0])
    >>> estimator.score(X, y)
    0.0
    """

    # This is a dictionary allowing to define the type of parameters.
    # It's used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "n_iter": [int],
        "batch_size": [int],
        "random_state": [int],
        "model_filename": [str],
        "lr": [float],
        "lr_patience": [int],
        "lr_reduction_factor": [float],
        "verbose": [bool],
        "skip_training": [bool],
        "save_trained": [bool],
        "storage_path": [str],
        "to_standardize": [str],
        "val_frac": [float],
    }

    def __init__(
        self,
        n_iter=10,
        batch_size=20,
        random_state=711,
        val_frac=0.1,
        lr=0.1,
        lr_patience=5,
        lr_reduction_factor=0.1,
        verbose=True,
        skip_training=True,
        save_trained=True,
        model_filename="_laplace_base.pth",
        storage_path="comparison_storage",
        to_standardize='X',
    ):
        """
        :param n_iter: 
        :param batch_size:
        :param random_state: 
        :param model_filename: 
        :param lr: 
        :param lr_patience: 
        :param lr_reduction_factor: 
        :param verbose: 
        :param skip_training: 
        :param save_trained: 
        :param storage_path: 
        :param to_standardize: 
        """
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.val_frac = val_frac
        self.model_filename = model_filename
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_reduction_factor = lr_reduction_factor
        self.verbose = verbose
        self.skip_training = skip_training
        self.save_trained = save_trained
        # todo: bad/inappropriate params to pass (not refering to algorithm)?
        self.storage_path = storage_path
        self.to_standardize = to_standardize

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        # todo: add model_params_choices=None,
        """A reference implementation of a fitting function.

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
        X, y = self._validate_data(X, y, accept_sparse=False)

        ##########
        io_helper = IO_Helper(self.storage_path)

        # if skip_training:
        #     print("skipping base model training")
        #     try:
        #         model = self.io_helper.load_model(model_filename)
        #         model.eval()
        #         return model
        #     except FileNotFoundError:
        #         # fmt: off
        #         print("error. model not found, so training cannot be skipped. training from scratch")

        torch.manual_seed(self.random_state)

        if len(y.shape) < 2:
            y = y.reshape(-1, 1)

        try:
            X_train, y_train = map(lambda arr: self._arr_to_tensor(arr), (X, y))
        except TypeError:
            raise TypeError(f'Unknown label type: {X.dtype} (X) or {y.dtype} (y)')

        n_samples = X_train.shape[0]
        val_size = max(1, round(self.val_frac * n_samples))
        train_size = max(1, n_samples-val_size)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train, y_train = X_train[:train_size], y_train[:train_size]

        assert X_train.shape[0] > 0 and X_val.shape[0] > 0

        dim_in, dim_out = X_train.shape[-1], y_train.shape[-1]

        model = self._nn_builder(
            dim_in,
            dim_out,
            num_hidden_layers=2,
            hidden_layer_size=50,
            # activation=torch.nn.LeakyReLU,
        )

        train_loader = self._get_train_loader(X_train, y_train, self.batch_size)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=self.lr_patience, factor=self.lr_reduction_factor
        )

        train_losses, val_losses = [], []
        iterable = tqdm(range(self.n_iter)) if self.verbose else range(self.n_iter)
        for _ in iterable:
            model.train()
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = self._mse_torch(model(X_val), y_val)
                train_loss = self._mse_torch(
                    model(X_train[:val_size]), y_val[:val_size]
                )
            scheduler.step(val_loss)
            val_losses.append(val_loss)
            train_losses.append(train_loss)

        self.model_ = model

        if self.save_trained:
            model_savepath = io_helper.get_model_savepath(self.model_filename)
            torch.save(model, model_savepath)

        if self.verbose:
            loss_skip = 0
            self._plot_training_progress(
                train_losses[loss_skip:], val_losses[loss_skip:]
            )

        ##########

        self.is_fitted_ = True
        # `fit` should always return `self`
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
    def _arr_to_tensor(arr) -> torch.Tensor:
        return torch.Tensor(arr).float()

    @staticmethod
    def _plot_training_progress(train_losses, test_losses):
        fig, ax = plt.subplots()
        ax.semilogy(train_losses, label="train")
        ax.semilogy(test_losses, label="val")
        ax.legend()
        plt.show()

    @staticmethod
    def _mse_torch(y_pred, y_test):
        return torch.mean((y_pred - y_test) ** 2)

    def predict(self, X):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=True, reset=False)

        # todo

        return np.ones(X.shape[0], dtype=np.int64)

    def _more_tags(self):
        return {'poor_score': True}


if __name__ == '__main__':
    estimator = MyEstimator(verbose=False)
    check_estimator(estimator)
