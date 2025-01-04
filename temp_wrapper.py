from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from helpers import misc_helpers


FILEPATH = './data/data_1600.pkl'
PLOT = True
OUTPUT_DIMS = [1, 2]


class Wrapper:
    def __init__(self, model, output_dim: Literal[1, 2] = 2):
        self.model_ = model
        self.output_dim = output_dim

    def predict(self, input_):
        output = self.model_.predict(input_)
        if self.output_dim == 1:
            output = misc_helpers.make_y_1d(output)
        else:
            output = misc_helpers.make_arr_2d(output)
        return output

    def set_output_dim(self, output_dim):
        self.output_dim = output_dim

    def __call__(self, input_):
        return self.predict(input_)

    def __getattr__(self, item):
        model = self.__getattribute__('model_')  # workaround bcdirect attr access doesn't work
        return getattr(model, item)

    def __setattr__(self, key, value):
        self.__dict__[key] = value


def posthoc_conformal_prediction(
        X_train: 'np.ndarray',
        y_train: 'np.ndarray',
        X_val: 'np.ndarray',
        y_val: 'np.ndarray',
        X_pred: 'np.ndarray',
        quantiles: list,
        base_model_wrapped,
        n_estimators=3,
        output_dim=2,
):
    from src_uq_methods_posthoc.conformal_prediction import (
        train_conformal_prediction,
        predict_with_conformal_prediction,
    )
    base_model_wrapped.set_output_dim(output_dim)
    model = train_conformal_prediction(
        X_train,
        y_train,
        X_val,
        y_val,
        base_model_wrapped,
        n_estimators=n_estimators,
        bootstrap_n_blocks=10,
        bootstrap_overlapping_blocks=False,
        random_seed=42,
        verbose=1,
    )
    # noinspection PyUnboundLocalVariable
    y_pred, y_quantiles, y_std = predict_with_conformal_prediction(model, X_pred, quantiles)
    base_model_wrapped.set_output_dim(2)
    return y_pred, y_quantiles, y_std


def main():
    print('start')
    quantiles = [0.1, 0.5, 0.9]
    print('load data')
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers.get_data(
        FILEPATH, n_points_per_group=800
    )
    print('fit model')
    base_model_wrapped = Wrapper(linear_model.LinearRegression())
    base_model_wrapped.fit(X_train, y_train)
    print('posthoc CP')
    for output_dim in OUTPUT_DIMS:
        print(f'running for output_dim={output_dim}...')
        y_pred, y_quantiles, y_std = posthoc_conformal_prediction(
            X_train,
            y_train,
            X_val,
            y_val,
            X,
            quantiles,
            base_model_wrapped,
        )
        arrs = {
            'X_train': X_train,
            'y_train': y_train,
            'y_pred': y_pred,
            'y_quantiles': y_quantiles,
            'y_std': y_std,
        }
        for arr_name, arr in arrs.items():
            print(arr_name)
            print('\tshape:', arr.shape)
            print('\tcontent', arr[:5])
        if PLOT:
            plot(X, y, y_pred, y_quantiles, filename=f'temp_plot_od{output_dim}.png')
        print(f'done with output_dim={output_dim}.')
    print('end')


def plot(X, y, y_pred, y_quantiles, filename):
    print('plot')
    x_plot = np.arange(X.shape[0])
    ci_low, ci_high = y_quantiles[:, 0], y_quantiles[:, -1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    ax.plot(x_plot, y, label='y true')
    ax.plot(x_plot, y_pred, label='y pred')
    ax.fill_between(
        x_plot.ravel(),
        ci_low,
        ci_high,
        color="green",
        alpha=0.2,
        label='quantiles',
    )
    ax.legend()
    plt.savefig(filename)
    plt.show(block=True)


if __name__ == '__main__':
    main()
