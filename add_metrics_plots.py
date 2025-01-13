import seaborn as sns
from matplotlib import pyplot as plt

from helpers.uq_arr_helpers import get_uq_method_to_arrs_gen


RUN_SIZE = 'small'


def main():
    from helpers.misc_helpers import _quick_load_data
    from helpers.io_helper import IO_Helper
    from helpers import misc_helpers

    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = _quick_load_data(RUN_SIZE)
    X_train, y_train = misc_helpers.add_val_to_train(X_train, X_val, y_train, y_val)
    n_samples_train = X_train.shape[0]

    io_helper_small = IO_Helper(arrays_folder='arrays_small', models_folder='models_small')
    uq_method_to_arrs_gen = get_uq_method_to_arrs_gen(io_helper=io_helper_small)
    for uq_method, uq_arrs in uq_method_to_arrs_gen:
        arrs_train, arrs_test = split_pred_arrs_train_test(uq_arrs, n_samples_train=n_samples_train)
        for y_true, arrs, are_train_arrs in [(y_train, arrs_train, True), (y_test, arrs_test, False)]:
            y_pred, y_quantiles, y_std = arrs
            plot_hist_crps(y_true, y_quantiles)


def split_pred_arrs_train_test(arrs, n_samples_train):
    arrs_train = list(map(lambda arr: arr[:n_samples_train], arrs))
    arrs_test = list(map(lambda arr: arr[n_samples_train:], arrs))
    return arrs_train, arrs_test


def plot_hist_crps(y_true, y_quantiles, bins=25, stat='density'):
    from helpers._metrics import crps
    score = crps(y_true, y_quantiles, keep_dim=True)
    sns.displot(score, bins=bins, stat=stat)
    # todo: add labels
    _show()


def _show():
    plt.show(block=True)


if __name__ == '__main__':
    main()
