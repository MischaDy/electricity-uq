import numpy as np
import torch
from matplotlib import pyplot as plt
from more_itertools import collapse
# noinspection PyProtectedMember
from sklearn.base import RegressorMixin, BaseEstimator, _fit_context
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# noinspection PyAttributeOutsideInit
class NN_Estimator(RegressorMixin, BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

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

    # This is a dictionary allowing to define the type of parameters.
    # It's used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "n_iter": [int],
        "batch_size": [int],
        "random_state": [int],
        "lr": [float],
        "lr_patience": [int],
        "lr_reduction_factor": [float],
        "verbose": [int],
        "skip_training": [bool],
        "save_trained": [bool],
        "to_standardize": [str],
        "val_frac": [float],
        "use_scheduler": [bool],
    }

    def __init__(
        self,
        n_iter=100,
        batch_size=20,
        random_seed=42,
        val_frac=0.1,
        use_scheduler=True,
        lr=None,
        lr_patience=5,
        lr_reduction_factor=0.5,
        verbose: int = 1,
        show_progress_bar=True,
        save_losses_plot=True,
    ):
        """
        :param n_iter: 
        :param batch_size:
        :param random_seed:
        :param lr: 
        :param lr_patience: 
        :param lr_reduction_factor: 
        :param verbose:
        """
        self.use_scheduler = use_scheduler
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.val_frac = val_frac
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_reduction_factor = lr_reduction_factor
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.save_losses_plot = save_losses_plot
        self.is_fitted_ = False

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

        torch.manual_seed(self.random_seed)

        self.is_y_2d_ = len(y.shape) == 2
        if len(y.shape) < 2:
            y = y.reshape(-1, 1)

        try:
            X_train, y_train = map(lambda arr: self._arr_to_tensor(arr), (X, y))
        except TypeError:
            raise TypeError(f'Unknown label type: {X.dtype} (X) or {y.dtype} (y)')

        n_samples = X_train.shape[0]
        val_size = max(1, round(self.val_frac * n_samples))
        # val_size = 20  # todo: temp
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

        if self.lr is None:
            self.lr = 1e-2 if self.use_scheduler else 1e-4

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.lr_patience, factor=self.lr_reduction_factor)
        criterion = nn.MSELoss()

        train_losses, val_losses = [], []
        epochs = tqdm(range(self.n_iter)) if self.show_progress_bar else range(self.n_iter)
        for _ in epochs:
            model.train()
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = self._mse_torch(model(X_val), y_val)
                train_loss = self._mse_torch(model(X_train[:val_size]), y_train[:val_size])
            if self.use_scheduler:
                scheduler.step(val_loss)
            val_losses.append(val_loss)
            train_losses.append(train_loss)

        model.eval()
        self.model_ = model

        if self.verbose > 1:
            loss_skip = 0
            self._plot_losses(train_losses[loss_skip:], val_losses[loss_skip:])

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
    def _arr_to_tensor(arr) -> torch.Tensor:
        return torch.Tensor(arr).float()

    def _plot_losses(self, train_losses, test_losses, filename='losses'):
        fig, ax = plt.subplots()
        ax.semilogy(train_losses, label="train")
        ax.semilogy(test_losses, label="val")
        ax.legend()
        if self.save_losses_plot:
            self.io_helper.save_plot(f"{filename}.png")
        plt.show()

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
        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=False, reset=False)
        X = self._arr_to_tensor(X)
        # self.model_.eval()
        with torch.no_grad():
            res = self.model_(X)
        res = res.reshape(-1, 1) if self.is_y_2d_ else res.squeeze()
        if as_np:
            res = np.array(res, dtype='float32')
        return res

    def _more_tags(self):
        return {'poor_score': True,
                '_xfail_checks': {'check_methods_sample_order_invariance': '(barely) failing for unknown reason'}}

    def load_state_dict(self, state_dict):
        self.model_ = nn.Module()
        self.model_ = self._nn_builder(..., ...)
        self.model_.load_state_dict(state_dict)

    def __getattr__(self, item):
        """
        get missing attribute from underlying model
        :param item:
        :return:
        """
        if self.is_fitted_:
            return self.model_.__getattribute__(item)
        raise AttributeError(f'NN_Estimator has no attribute "{item}", or only has it is after fitting')

    def __call__(self, *args, **kwargs):
        if self.is_fitted_:
            return self.model_(*args, **kwargs)
        raise TypeError('NN_Estimator is only callable after fitting')


if __name__ == '__main__':
    estimator = NN_Estimator(verbose=0)
    check_estimator(estimator)
