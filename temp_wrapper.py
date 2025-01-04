import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from helpers import misc_helpers


class Wrapper:
    def __init__(self, model):
        self.model_ = model

    def __getattr__(self, item):
        return getattr(self.__getattribute__('model_'), item)

    def __call__(self, input_):
        return self.model_(input_)

    def __setattr__(self, key, value):
        self.__dict__[key] = value


def main():
    print('start')
    print('load data')
    filepath = './data/data_2015_2018.pkl'
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers.get_data(filepath,
                                                                                           n_points_per_group=800)
    print('fit model')
    wrapper = Wrapper(linear_model.LinearRegression())
    wrapper.fit(X_train, y_train)
    print('predict')
    y_pred = wrapper.predict(X)
    print('plot')
    x_plot = np.arange(X.shape[0])
    plt.plot(x_plot, y_train, label='y train')
    plt.plot(x_plot, y_pred, label='y pred')
    plt.legend()
    plt.show(block=True)
    print('end')


if __name__ == '__main__':
    main()
