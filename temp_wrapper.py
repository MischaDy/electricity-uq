from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from helpers import misc_helpers


FILEPATH = './data/data_1600.pkl'
PLOT = True
OUTPUT_DIM = 1


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


def main():
    print('start')
    print('load data')
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers.get_data(
        FILEPATH, n_points_per_group=800
    )
    print('fit model')
    wrapper = Wrapper(linear_model.LinearRegression())
    wrapper.fit(X_train, y_train)
    print('predict')
    wrapper.set_output_dim(OUTPUT_DIM)
    y_pred = wrapper.predict(X)
    print('shapes:', X_train.shape, y_train.shape, y_pred.shape)
    if PLOT:
        plot(X, y, y_pred)
    print('end')


def plot(X, y, y_pred):
    print('plot')
    x_plot = np.arange(X.shape[0])
    plt.plot(x_plot, y, label='y true')
    plt.plot(x_plot, y_pred, label='y pred')
    plt.legend()
    plt.show(block=True)


if __name__ == '__main__':
    main()
