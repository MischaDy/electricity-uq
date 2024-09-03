import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def get_data(n_points_temp, filepath='data.pkl', input_cols=None, output_cols=None, return_full_data=False):
    """load and prepare data"""
    if input_cols is None:
        input_cols = [
            'load_last_week',
            'load_last_hour',
            'load_now',
            'is_workday',
            'is_saturday_and_not_holiday',
            'is_sunday_or_holiday',
            'is_heating_period',
        ]
    if output_cols is None:
        output_cols = ['load_next_hour']
    df = pd.read_pickle(filepath)

    mid = df.shape[0] // 2
    X = df[input_cols].iloc[mid - n_points_temp: mid + n_points_temp]
    y = df[output_cols].iloc[mid - n_points_temp: mid + n_points_temp]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)
    if return_full_data:
        return X_train, X_test, y_train, y_test, X, y
    return X_train, X_test, y_train, y_test


def plot_data(X_train, X_test, y_train, y_test, filename='data', do_save_figure=False):
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
        IO_HELPER.save_plot(f'{filename}_{N_POINTS_TEMP}.png')
    plt.show()


def unzip(iterable):
    return np.array(list(zip(*iterable)))


class IOHelper:
    def __init__(self, base_folder, arrays_folder='arrays', models_folder='models', plots_folder='plots'):
        # (self.arrays_folder,
        #  self.models_folder,
        #  self.plots_folder) = (os.path.join(base_folder, folder_name)
        #                        for folder_name in (arrays_folder, models_folder, plots_folder))
        self.arrays_folder = os.path.join(base_folder, arrays_folder)
        self.models_folder = os.path.join(base_folder, models_folder)
        self.plots_folder = os.path.join(base_folder, plots_folder)
        self.folders = [self.arrays_folder, self.models_folder, self.plots_folder]
        for folder in self.folders:
            try: os.mkdir(folder)
            except FileExistsError: pass

    def get_array_savepath(self, filename):
        return os.path.join(self.arrays_folder, filename)

    def get_model_savepath(self, filename):
        return os.path.join(self.models_folder, filename)

    def get_plot_savepath(self, filename):
        return os.path.join(self.plots_folder, filename)

    def load_array(self, filename):
        return np.load(self.get_array_savepath(filename))

    def load_model(self, filename):
        return pickle.load(open(self.get_model_savepath(filename), 'rb'))

    def save_array(self, array, filename):
        np.save(self.get_array_savepath(filename), array)

    def save_model(self, model, filename):
        pickle.dump(model, open(self.get_model_savepath(filename), 'wb'))

    def save_plot(self, filename):
        plt.savefig(self.get_plot_savepath(filename))
