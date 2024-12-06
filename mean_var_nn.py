import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from matplotlib import pyplot as plt

from sklearn.base import RegressorMixin, BaseEstimator, _fit_context
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from more_itertools import collapse
from tqdm import tqdm
from uncertainty_toolbox import nll_gaussian


class MeanVarNN(nn.Module):
    def __init__(
        self,
        dim_in,
        num_hidden_layers=2,
        hidden_layer_size=50,
        activation=torch.nn.LeakyReLU,
    ):
        super().__init__()
        layers = collapse([
            torch.nn.Linear(dim_in, hidden_layer_size),
            activation(),
            [[torch.nn.Linear(hidden_layer_size, hidden_layer_size),
              activation()]
             for _ in range(num_hidden_layers)],
            torch.nn.Linear(hidden_layer_size, 2),
        ])
        self.first_layer_stack = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_layer_stack(x)
        x = self.output_activation(x)
        return x

    @staticmethod
    def output_activation(x) -> tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: tensor of shape (n_samples, 2), where the dimension 1 are means and dimension 2 are variances
        :return:
        """
        mean, var = x[:, 0], x[:, 1]
        var = F.relu(var)
        return mean, var  # todo: possible?


# noinspection PyAttributeOutsideInit
class MeanVarNN_Estimator(RegressorMixin, BaseEstimator):
    # This is a dictionary allowing to define the type of parameters.
    # It's used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "n_iter": [int],
        "batch_size": [int],
        "random_state": [int],
        "lr": [float],
        "lr_patience": [int],
        "lr_reduction_factor": [float],
        "verbose": [bool],
        "skip_training": [bool],
        "save_trained": [bool],
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
            lr_reduction_factor=0.5,
            verbose=True,
    ):
        """
        :param n_iter:
        :param batch_size:
        :param random_state:
        :param lr:
        :param lr_patience:
        :param lr_reduction_factor:
        :param verbose:
        """
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.val_frac = val_frac
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_reduction_factor = lr_reduction_factor
        self.verbose = verbose

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

        torch.manual_seed(self.random_state)

        self.is_y_2d_ = len(y.shape) == 2
        if len(y.shape) < 2:
            y = y.reshape(-1, 1)

        try:
            X_train, y_train = map(lambda arr: self._arr_to_tensor(arr), (X, y))
        except TypeError:
            raise TypeError(f'Unknown label type: {X.dtype} (X) or {y.dtype} (y)')

        n_samples = X_train.shape[0]
        val_size = max(1, round(self.val_frac * n_samples))
        train_size = max(1, n_samples - val_size)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train, y_train = X_train[:train_size], y_train[:train_size]

        assert X_train.shape[0] > 0 and X_val.shape[0] > 0

        dim_in, dim_out = X_train.shape[-1], y_train.shape[-1]

        model = MeanVarNN(
            dim_in,
            num_hidden_layers=2,
            hidden_layer_size=50,
        )

        train_loader = self._get_train_loader(X_train, y_train, self.batch_size)

        criterion = nn.GaussianNLLLoss()
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
                y_pred_mean, y_pred_var = model(X)
                loss = criterion(y_pred_mean, y, y_pred_var)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = self._nll_loss(model(X_val), y_val)
                train_loss = self._nll_loss(model(X_train[:val_size]), y_train[:val_size])
            scheduler.step(val_loss)
            val_losses.append(val_loss)
            train_losses.append(train_loss)

        model.eval()
        self.model_ = model

        if self.verbose:
            loss_skip = 0
            self._plot_training_progress(
                train_losses[loss_skip:], val_losses[loss_skip:]
            )

        ##########

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

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
    def _nll_loss(y_pred, y_test):
        y_pred_mean, y_pred_var = y_pred
        return nll_gaussian(y_pred_mean, y_test, np.sqrt(y_pred_var))

    def predict(self, X, as_np=True):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
            :param as_np:
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
        res = res.reshape(-1, 1) if self.is_y_2d_ else res.ravel()
        if as_np:
            res = np.array(res, dtype='float32')
        return res

    def _more_tags(self):
        return {'poor_score': True,
                '_xfail_checks': {'check_methods_sample_order_invariance': '(barely) failing for unknown reason'}}

    def eval(self):
        self.model_.eval()


def native_mean_var_nn(
        X_train,
        y_train,
        n_iter=100,
        batch_size=20,
        random_state=711,
        verbose=True,
        val_frac=0.1,
        lr=0.1,
        lr_patience=5,
        lr_reduction_factor=0.5,
):
    model = MeanVarNN_Estimator(
        n_iter=n_iter,
        batch_size=batch_size,
        random_state=random_state,
        val_frac=val_frac,
        lr=lr,
        lr_patience=lr_patience,
        lr_reduction_factor=lr_reduction_factor,
        verbose=verbose,
    )
    model.fit(X_train, y_train)

    # if verbose:
    #     plot_post_training_perf(model, X_train, y_train, do_save_figure=True)
    return model


if __name__ == '__main__':
    estimator = MeanVarNN_Estimator(verbose=False)
    check_estimator(estimator)
