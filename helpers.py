import numpy as np
import torch


def get_data(
    _n_points_per_group,
    filepath="data.pkl",
    input_cols=None,
    output_cols=None,
    return_full_data=False,
):
    # -> (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    #   | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]):  # fmt: skip
    """
    load and prepare data

    :param _n_points_per_group:
    :param filepath:
    :param input_cols:
    :param output_cols:
    :param return_full_data:
    :return: tuple (X_train, X_test, y_train, y_test), or (..., X, y) if return_full_data is True
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

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


def plot_data(X_train, X_test, y_train, y_test, io_helper=None, filename="data", do_save_figure=False):
    """visualize training and test sets"""
    from matplotlib import pyplot as plt

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
    from more_itertools import collapse
    arr = list(collapse(arrays))
    return all(a <= b for a, b in zip(arr, arr[1:]))


def standardize(train_data, *arrays_to_standardize, return_scaler=False):
    from sklearn.preprocessing import StandardScaler

    # todo: bugfix - only standardize continuous columns!
    scaler = StandardScaler()
    scaler.fit(train_data)

    def transform(arr):
        if arr is None:
            return
        return scaler.transform(arr)

    standardized_data = map(transform, [train_data, *arrays_to_standardize])
    return standardized_data if not return_scaler else (scaler, standardized_data)


def allow_none(func):
    def wrapper(arg, **kwargs):
        if arg is None:
            return
        return func(arg, **kwargs)
    return wrapper


@allow_none
def df_to_numpy(df):  #: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=float)


@allow_none
def df_to_tensor(df):  # pd.DataFrame) -> torch.Tensor:
    return numpy_to_tensor(df_to_numpy(df))


@allow_none
def numpy_to_tensor(arr: np.ndarray):  # -> torch.Tensor:
    return torch.Tensor(arr).float()


@allow_none
def tensor_to_numpy(tensor: torch.Tensor):  # -> np.ndarray:
    return tensor.numpy(force=True)


def tensor_to_device(tensor: torch.Tensor) -> torch.Tensor:
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print('warning: cuda not available!')
        return tensor
    device = torch.cuda.current_device()
    return tensor.to(device)


def make_arr_2d(arr):
    return arr.reshape(-1, 1)


def get_train_loader(X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int):
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader
