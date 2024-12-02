import os
import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from more_itertools import flatten, collapse
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


def get_data(
    _n_points_per_group,
    filepath="data.pkl",
    input_cols=None,
    output_cols=None,
    return_full_data=False,
) -> (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
      | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]):  # fmt: skip
    """
    load and prepare data

    :param _n_points_per_group:
    :param filepath:
    :param input_cols:
    :param output_cols:
    :param return_full_data:
    :return: tuple (X_train, X_test, y_train, y_test), or (..., X, y) if return_full_data is True
    """
    df = pd.read_pickle(filepath)
    if output_cols is None:
        output_cols = ["load_to_pred"]
    if input_cols is None:
        # input_cols = [
        #     "load_last_week",
        #     "load_last_hour",
        #     "load_now",
        #     "is_workday",
        #     "is_saturday_and_not_holiday",
        #     "is_sunday_or_holiday",
        #     "is_heating_period",
        # ]
        input_cols = [col for col in df.columns
                      if col not in output_cols and not col.startswith('ts')]

    # mid = df.shape[0] // 2
    # X = df[input_cols].iloc[mid - _n_points_per_group: mid + _n_points_per_group]
    # y = df[output_cols].iloc[mid - _n_points_per_group: mid + _n_points_per_group]

    lim = 2*_n_points_per_group if _n_points_per_group is not None else -1
    X = df[input_cols].iloc[:lim]
    y = df[output_cols].iloc[:lim]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )
    X_train, X_test, y_train, y_test, X, y = set_dtype_float(X_train, X_test, y_train, y_test, X, y)
    if return_full_data:
        return X_train, X_test, y_train, y_test, X, y
    return X_train, X_test, y_train, y_test


def set_dtype_float(*arrs):
    return map(lambda arr: arr.astype('float32'), arrs)


def plot_data(X_train, X_test, y_train, y_test, io_helper: "IO_Helper", filename="data", do_save_figure=False):
    """visualize training and test sets"""
    num_train_steps = X_train.shape[0]
    num_test_steps = X_test.shape[0]

    x_plot_train = np.arange(num_train_steps)
    x_plot_test = x_plot_train + num_test_steps

    plt.figure(figsize=(16, 5))
    plt.plot(x_plot_train, y_train)
    plt.plot(x_plot_test, y_test)
    plt.ylabel("energy data (details TODO)")
    plt.legend(["Training data", "Test data"])
    if do_save_figure:
        io_helper.save_plot(f"{filename}.png")
    plt.show()


def unzip(iterable):
    return np.array(list(zip(*iterable)))


# noinspection PyPep8Naming
class IO_Helper:
    def __init__(
        self,
        base_folder,
        arrays_folder="arrays",
        models_folder="models",
        plots_folder="plots",
    ):
        self.arrays_folder = os.path.join(base_folder, arrays_folder)
        self.models_folder = os.path.join(base_folder, models_folder)
        self.plots_folder = os.path.join(base_folder, plots_folder)
        self.folders = [self.arrays_folder, self.models_folder, self.plots_folder]
        for folder in self.folders:
            os.makedirs(folder, exist_ok=True)

    def get_array_savepath(self, filename):
        return os.path.join(self.arrays_folder, filename)

    def get_model_savepath(self, filename):
        return os.path.join(self.models_folder, filename)

    def get_plot_savepath(self, filename):
        return os.path.join(self.plots_folder, filename)

    def load_array(self, filename):
        return np.load(self.get_array_savepath(filename))

    def load_model(self, filename):
        with open(self.get_model_savepath(filename), "rb") as file:
            model = pickle.load(file)
        return model

    def load_torch_model(self, filename, *args, **kwargs):
        """

        :param filename:
        :param args: args for torch.load
        :param kwargs: kwargs for torch.load
        :return:
        """
        # todo: correct?
        filepath = self.get_model_savepath(filename)
        kwargs['weights_only'] = True
        return torch.load(filepath, *args, **kwargs)

    def save_array(self, array, filename):
        np.save(self.get_array_savepath(filename), array)

    def save_model(self, model, filename):
        with open(self.get_model_savepath(filename), "wb") as file:
            pickle.dump(model, file)

    def save_plot(self, filename):
        plt.savefig(self.get_plot_savepath(filename))


def check_is_ordered(pred, pis):
    lower_ordered = np.all(pis[:, 0] > pred)
    higher_ordered = np.all(pred <= pis[:, 1])
    if not lower_ordered:
        print("not lower-ordered!")
    if not higher_ordered:
        print("not higher-ordered!")
    if lower_ordered and higher_ordered:
        print("ordered.")
        return True
    return False


def starfilter(pred, iterable):
    return filter(lambda args: pred(*args), iterable)


def identity(*args):
    if len(args) > 2:
        res = args
    elif len(args) == 1:
        res = args[0]
    else:
        res = None
    return res


def is_ascending(*arrays):
    """

    :param arrays:
    :return: whether the (fully!) flattened concat of the arrays is ascending
    """
    # todo: generalize to is_ordered
    arr = list(collapse(arrays))
    return all(a <= b for a, b in zip(arr, arr[1:]))


def standardize(return_scaler, train_data, *arrays_to_standardize):
    # todo: bugfix - only standardize continuous columns!
    scaler = StandardScaler()
    scaler.fit(train_data)
    standardized_data = map(scaler.transform, [train_data, *arrays_to_standardize])
    return standardized_data if not return_scaler else (scaler, standardized_data)