from typing import Generator, Any, TYPE_CHECKING

import numpy as np
import logging

if TYPE_CHECKING:
    import pandas as pd
    import torch
    from helpers.io_helper import IO_Helper
    from helpers.typing_ import TrainValTestDataFrames


def get_data(
        filepath,
        train_years: tuple[int, int] = None,
        val_years: tuple[int, int] = None,
        test_years: tuple[int, int] = None,
        n_points_per_group=None,
        input_cols=None,
        output_cols=None,
        do_standardize_data=True,
):
    """
    load and prepare data

    :param test_years:
    :param val_years:
    :param train_years:
    :param do_standardize_data:
    :param n_points_per_group:
    :param filepath:
    :param input_cols:
    :param output_cols:
    :return:
    A tuple (X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y).
    If standardize_data=False, y_scaler is None. All variables except for the scaler are 2D np arrays.
    """
    X, y, numerical_cols = load_data(
        filepath,
        input_cols=input_cols,
        output_cols=output_cols,
        do_output_numerical_col_names=True,
        n_points_per_group=n_points_per_group,
        return_ts_col=True,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, train_years=train_years, val_years=val_years, test_years=test_years, val_size=0.1,
    )
    if do_standardize_data:
        X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = standardize_data(
            X_train, y_train, X_val, y_val, X_test, y_test, X, y, numerical_cols=numerical_cols
        )
    else:
        scaler_y = None

    # todo: where does casting to arrays happen? can make it happen earlier?
    # to float arrays
    X_train, y_train, X_val, y_val, X_test, y_test, X, y = set_dtype_float(
        X_train, y_train, X_val, y_val, X_test, y_test, X, y
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y


def standardize_data(X_train, y_train, X_val, y_val, X_test, y_test, X, y, numerical_cols=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import make_column_transformer

    if numerical_cols is None:
        numerical_cols = []

    # standardize X
    scaler_X = make_column_transformer(
        (StandardScaler(), numerical_cols),
        remainder='passthrough',
        force_int_remainder_cols=False,
    )
    scaler_X.fit(X_train)
    X_train, X_val, X_test, X = map(scaler_X.transform, [X_train, X_val, X_test, X])

    # standardize y
    scaler_y = StandardScaler()
    scaler_y.fit(y_train)
    y_train, y_val, y_test, y = map(scaler_y.transform, [y_train, y_val, y_test, y])
    return X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y


def train_val_test_split(
        X: 'pd.DataFrame',
        y: 'pd.DataFrame',
        train_years: tuple[int, int] = None,
        val_years: tuple[int, int] = None,
        test_years: tuple[int, int] = None,
        train_size: float = 0.4,
        val_size: float = 0.1,
):
    """
    split data into train/val/test sets. Any ts cols present in X will be dropped.

    :param X:
    :param y:
    :param train_years:
    :param val_years:
    :param test_years:
    :param train_size: size of train set as proportion of total data (excluding validation set)
    :param val_size: size of validation set as proportion of total data
    :return: tuple (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    years_ranges = [train_years, val_years, test_years]
    if None not in years_ranges:
        X_train, X_val, X_test, y_train, y_val, y_test = _train_val_test_split_by_year(X, y, years_ranges)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = _train_val_test_split_by_size(X, y, train_size, val_size)
    for arr in [X_train, X_test, y_train, y_test]:
        ts_cols = [col for col in arr.columns if col.startswith('ts_')]
        arr.drop(columns=ts_cols, inplace=True)  # todo: works bc no ts cols present?
    return X_train, y_train, X_val, y_val, X_test, y_test


def _train_val_test_split_by_size(
        X: 'pd.DataFrame', y: 'pd.DataFrame', train_size: float = 0.5, val_size: float = 0.1
) -> 'TrainValTestDataFrames':
    import pandas as pd
    pd.options.mode.copy_on_write = True

    n_samples = X.shape[0]
    cutoff_train_val = round(train_size * n_samples)
    cutoff_val_test = cutoff_train_val + round(val_size * n_samples)
    cutoffs = [0, cutoff_train_val, cutoff_val_test, None]
    (X_train, X_val, X_test), (y_train, y_val, y_test) = (
        [arr[from_:to]
         for from_, to in zip(cutoffs, cutoffs[1:])]
        for arr in (X, y)
    )
    # noinspection PyTypeChecker
    return X_train, X_val, X_test, y_train, y_val, y_test


def _train_val_test_split_by_year(
        X: 'pd.DataFrame', y: 'pd.DataFrame', years_ranges: list[tuple[int, int]]
) -> 'TrainValTestDataFrames':
    assert len(years_ranges) == 3
    X_years = X.ts_pred.map(lambda ts: ts.year)
    years_indices_all = (_get_years_indices(X_years, years_range)
                         for years_range in years_ranges)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = (
        (X[years_indices], y[years_indices])
        for years_indices in years_indices_all
    )
    # noinspection PyTypeChecker
    return X_train, X_val, X_test, y_train, y_val, y_test


def _get_years_indices(X_years: 'pd.Series', years_range: tuple[int, int]) -> 'pd.Series':
    """

    :param X_years:
    :param years_range:
    :return: indices of X_years where each year is in years_range, excluding endpoint
    """
    # noinspection PyTypeChecker
    return (years_range[0] <= X_years) & (X_years < years_range[1])


def load_data(filepath, input_cols=None, output_cols=None, do_output_numerical_col_names=True, n_points_per_group=None,
              return_ts_col=False):
    import pandas as pd

    df = pd.read_pickle(filepath)
    if output_cols is None:
        output_cols = ["load_to_pred"]
    if input_cols is None:
        # input_cols = ["load_last_week", "load_last_hour", "load_now", "cat_is_workday",
        # "cat_is_saturday_and_not_holiday", "cat_is_sunday_or_holiday", "cat_is_heating_period"]
        input_cols = [col for col in df.columns
                      if col not in output_cols]
        if not return_ts_col:
            input_cols = [col for col in input_cols if not col.startswith('ts_')]
    numerical_col_names = [col for col in input_cols
                           if not col.startswith('cat_') and not col.startswith('ts_')]
    lim = 2 * n_points_per_group if n_points_per_group is not None else None
    X = df[input_cols].iloc[:lim]
    y = df[output_cols].iloc[:lim]
    if do_output_numerical_col_names:
        return X, y, numerical_col_names
    return X, y


def inverse_transform_y(scaler_y, y: np.ndarray):
    n_dim = len(y.shape)
    if n_dim < 2:
        y = make_arr_2d(y)
    y = scaler_y.inverse_transform(y)
    if n_dim < 2:
        y = make_arr_1d(y)
    return y


def inverse_transform_ys(scaler_y, *ys: np.ndarray, to_np=False):
    transformed = map(lambda y: inverse_transform_y(scaler_y, y), ys)
    if not to_np:
        return transformed
    return np.array(list(transformed))


def upscale_y_std(scaler_y, y_std):
    return scaler_y.scale_ * y_std


def set_dtype_float(*arrs: list[np.ndarray]) -> Generator[np.ndarray, None, None]:
    yield from map(lambda arr: arr.astype('float32'), arrs)


def plot_data(X_train, X_test, y_train, y_test, io_helper: 'IO_Helper' = None, filename="data", do_save_figure=False):
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
        io_helper.save_plot(filename=filename)
    plt.show(block=True)


def unzip(iterable):
    return np.array(list(zip(*iterable)))


def check_is_ordered(pred, pis):
    lower_ordered = np.all(pis[:, 0] > pred)
    higher_ordered = np.all(pred <= pis[:, 1])
    if not lower_ordered:
        logging.warning("not lower-ordered!")
    if not higher_ordered:
        logging.warning("not higher-ordered!")
    if lower_ordered and higher_ordered:
        logging.info("ordered.")
        return True
    return False


def starfilter(pred, iterable):
    return filter(lambda args: pred(*args), iterable)


def identity(*args):
    if len(args) > 2:
        result = args
    elif len(args) == 1:
        result = args[0]
    else:
        result = None
    return result


def is_ascending(*arrays):
    """

    :param arrays:
    :return: whether the (fully) flattened concatenation of the arrays is ascending
    """
    from more_itertools import collapse
    arr = list(collapse(arrays))
    return all(a <= b for a, b in zip(arr, arr[1:]))


def df_to_np_array(df: 'pd.DataFrame') -> np.ndarray:
    return df.to_numpy(dtype=float)


def dfs_to_np_arrays(*dfs: 'pd.DataFrame'):
    return map(df_to_np_array, dfs)


def df_to_tensor(df: 'pd.DataFrame') -> 'torch.Tensor':
    return np_array_to_tensor(df_to_np_array(df))


def dfs_to_tensors(*dfs):
    return map(df_to_tensor, dfs)


def np_array_to_tensor(arr: np.ndarray) -> 'torch.Tensor':
    import torch
    return torch.Tensor(arr).float()


def np_arrays_to_tensors(*arrays):
    return map(np_array_to_tensor, arrays)


def tensor_to_np_array(tensor: 'torch.Tensor') -> np.ndarray:
    return tensor.numpy(force=True).astype('float32')


def tensors_to_np_arrays(*tensors):
    return map(tensor_to_np_array, tensors)


def get_device():
    import torch
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def object_to_cuda(obj):
    device = get_device()
    return obj.to(device)


def objects_to_cuda(*objs: Any) -> Generator[Any, None, None]:
    import torch
    if not torch.cuda.is_available():
        yield from objs
    else:
        yield from map(object_to_cuda, objs)


def make_arr_2d(arr: np.ndarray):
    return arr.reshape(-1, 1)


def make_arrs_2d(*arrs):
    return map(make_arr_2d, arrs)


def make_arr_1d(arr: np.ndarray):
    return arr.squeeze()


def make_arrs_1d(*arrs):
    return map(make_arr_1d, arrs)


def make_tensor_contiguous(tensor: 'torch.Tensor'):
    return tensor.contiguous()


def make_tensors_contiguous(*tensors: 'torch.Tensor'):
    return map(make_tensor_contiguous, tensors)


def preprocess_array_to_tensor(array: np.ndarray) -> 'torch.Tensor':
    array = np_array_to_tensor(array)
    array = object_to_cuda(array)
    array = make_tensor_contiguous(array)
    return array


def preprocess_arrays_to_tensors(*arrays: np.ndarray):
    return map(preprocess_array_to_tensor, arrays)


def get_train_loader(X_train: 'torch.Tensor', y_train: 'torch.Tensor', batch_size: int):
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader


def timestamped_filename(prefix, ext=None, randomize=False):
    filename = f'{prefix}_{get_timestamp()}'
    if randomize:
        postfix = get_random_string()
        filename = f'{filename}_{postfix}'
    if ext is not None:
        filename = f'{filename}.{ext}'
    return filename


def get_random_string(length=4):
    import random
    import string
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def get_timestamp():
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    return timestamp


def quantiles_gaussian(quantiles: list, y_pred: np.ndarray, y_std: np.ndarray):
    from scipy.stats import norm
    # todo: does this work for multi-dim outputs?
    return np.array([norm.ppf(quantiles, loc=mean, scale=std)
                     for mean, std in zip(y_pred, y_std)])


def stds_from_quantiles(quantiles: np.ndarray):
    """
    :param quantiles: array of shape (number of datapoints, number of quantiles), where number of quantiles should
    be at least about 100
    :return:
    """
    num_quantiles = quantiles.shape[1]
    if num_quantiles < 50:
        logging.warning(f"{num_quantiles} quantiles are too few to compute"
                        f" a reliable std from (should be about 100)")
    return np.std(quantiles, ddof=1, axis=1)


def pis_from_quantiles(quantiles):
    mid = len(quantiles) // 2
    first, second = quantiles[:mid], quantiles[mid:]
    pi_limits = zip(first, reversed(second))
    pis = [high - low for low, high in pi_limits]
    return sorted(pis)


def quantiles_from_pis(pis: np.ndarray, check_order=False):
    """
    currently "buggy" for odd number of quantiles.
    :param check_order:
    :param pis: prediction intervals array of shape (n_samples, 2, n_intervals)
    :return: array of quantiles of shape (n_samples, 2 * n_intervals)
    """
    if check_order:
        assert np.all([is_ascending(pi[0, :], reversed(pi[1, :])) for pi in pis])
    y_quantiles = np.array([sorted(pi.flatten()) for pi in pis])
    return y_quantiles


def measure_runtime(func):
    from timeit import default_timer

    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logging.info(f'running function {func_name} (with timing)...')
        t1 = default_timer()
        result = func(*args, **kwargs)
        t2 = default_timer()
        logging.info(f'function {func_name} is done. [took {t2 - t1}s]')
        return result

    return wrapper


def plot_nn_losses(
        train_losses,
        test_losses=None,
        show_plot=True,
        save_plot=False,
        io_helper: 'IO_Helper' = None,
        method_name=None,
        filename=None,
        postfix='losses',
        loss_skip=0,
):
    """
    preferably provide method_name over filename

    :param postfix:
    :param train_losses:
    :param test_losses:
    :param show_plot:
    :param save_plot:
    :param io_helper:
    :param method_name:
    :param filename:
    :param loss_skip:
    :return:
    """
    import matplotlib.pyplot as plt

    if not show_plot and not save_plot:
        logging.warning('nn loss plot to be neither shown nor saved. skipping entirely.')
        return

    if save_plot and io_helper is None:
        logging.error('io_helper must be set for nn loss to be saved!')
        save_plot = False

    fig, ax = plt.subplots()
    plt_func = ax.plot if _contains_neg(train_losses) or _contains_neg(test_losses) else ax.semilogy
    plt_func(train_losses[loss_skip:], label="train loss")
    plt_func(test_losses[loss_skip:], label="test loss")
    ax.legend()
    if save_plot:
        if method_name is not None:
            io_helper.save_plot(method_name=method_name, is_loss_plot=True)
        else:
            filename = postfix if filename is None else f'{filename}{io_helper.sep}{postfix}'
            io_helper.save_plot(filename=filename, is_loss_plot=True)
    if show_plot:
        plt.show(block=True)
    plt.close(fig)


def _contains_neg(losses):
    return any(map(lambda x: x < 0, losses))


def add_val_to_train(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray):
    X_train = np.vstack([X_train, X_val])
    y_train = np.vstack([y_train, y_val])
    return X_train, y_train


def build_unif_distr(start: float, end: float):
    from scipy import stats
    return stats.distributions.uniform(loc=start, scale=end - start)


def get_device_of_nn(nn):
    return next(nn.parameters()).device


def quantiles_dict_to_np_arr(quantiles_dict: dict[float, 'np.ndarray']):
    import numpy as np
    return np.array(list(quantiles_dict.values())).T


def _quick_load_data(run_size):
    import settings
    from settings_update import update_run_size_setup

    logging.info('loading data')
    update_run_size_setup(run_size)
    data = get_data(
        filepath=settings.DATA_FILEPATH,
        train_years=settings.TRAIN_YEARS,
        val_years=settings.VAL_YEARS,
        test_years=settings.TEST_YEARS,
        do_standardize_data=True,
    )
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = data
    y_train, y_val, y_test, y = map(scaler_y.inverse_transform, [y_train, y_val, y_test, y])
    return X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y
