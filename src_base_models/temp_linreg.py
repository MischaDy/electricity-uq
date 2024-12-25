import logging

import numpy as np
from matplotlib import pyplot as plt

from helpers.misc_helpers import get_data
from helpers.io_helper import IO_Helper

DATA_FILEPATH = '../data/data_1600.pkl'
N_POINTS_PER_GROUP = 800
STANDARDIZE_DATA = True
STORAGE_PATH = "../comparison_storage"
SKIP_TRAINING = True
SAVE_MODEL = True


def base_model_linreg(
        X_train: np.ndarray,
        y_train: np.ndarray,
        io_helper,
        n_jobs=-1,
        skip_training=True,
        save_model=True,
):
    from sklearn import linear_model

    filename_base_model = f"base_linreg.model"
    if skip_training:
        try:
            logging.info('skipping linreg base model training')
            model = io_helper.load_model(filename_base_model)
            return model
        except FileNotFoundError:
            logging.warning(f"trained base model '{filename_base_model}' not found. training from scratch.")
    model = linear_model.LinearRegression(n_jobs=n_jobs)
    model.fit(X_train, y_train)
    if save_model:
        logging.info('saving linreg base model...')
        io_helper.save_model(model, filename_base_model)
    return model


def main():
    logging.info('loading data')
    X_train, X_test, y_train, y_test, X, y, y_scaler = get_data(
        filepath=DATA_FILEPATH,
        n_points_per_group=N_POINTS_PER_GROUP,
        standardize_data=STANDARDIZE_DATA,
    )

    logging.info('training model')
    io_helper = IO_Helper(STORAGE_PATH)
    model = base_model_linreg(
        X_train,
        y_train,
        io_helper,
        skip_training=SKIP_TRAINING,
        save_model=SAVE_MODEL,
    )
    logging.info('predicting')
    y_pred = model.predict(X)

    x_plot_train = np.arange(X_train.shape[0])
    x_plot_test = x_plot_train + X_test.shape[0]
    x_plot_pred = np.arange(X.shape[0])

    logging.info('plotting')
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_plot_train, y_train, label='y train')
    ax.plot(x_plot_test, y_test, label='y test')
    ax.plot(x_plot_pred, y_pred, label='y pred')
    ax.legend()
    plt.show(block=True)


if __name__ == '__main__':
    main()
