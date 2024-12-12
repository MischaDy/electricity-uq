import os
import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from more_itertools import collapse
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


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


def plot_data(X_train, X_test, y_train, y_test, io_helper: "IO_Helper" = None, filename="data", do_save_figure=False):
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

    def load_torch_model(self, model_class, filename, *args, **kwargs):
        """

        :param model_class:
        :param filename:
        :param args: args for model_class constructor
        :param kwargs: kwargs for model_class constructor
        :return:
        """
        model = model_class(*args, **kwargs)
        filepath = self.get_model_savepath(filename)
        model.load_state_dict(torch.load(filepath, weights_only=True))
        model.eval()
        return model

    def load_torch_model2(self, filename, *args, **kwargs):
        """

        :param filename:
        :param args: args for torch.load
        :param kwargs: kwargs for torch.load
        :return:
        """
        filepath = self.get_model_savepath(filename)
        kwargs['weights_only'] = False
        model = torch.load(filepath, *args, **kwargs)
        model.eval()
        return model

    def save_array(self, array, filename):
        path = self.get_array_savepath(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, array)

    def save_model(self, model, filename):
        path = self.get_model_savepath(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(model, file)

    def save_torch_model(self, model, filename):
        path = self.get_model_savepath(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)

    def save_plot(self, filename):
        path = self.get_plot_savepath(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)


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


def df_to_numpy(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=float)


def df_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    return numpy_to_tensor(df_to_numpy(df))


def numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.Tensor(arr).float()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy(force=True)


def tensor_to_device(tensor: torch.Tensor) -> torch.Tensor:
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        return tensor
    device = torch.cuda.current_device()
    return tensor.to(device)


def make_arr_2d(arr):
    return arr.reshape(-1, 1)


def get_train_loader(X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader
