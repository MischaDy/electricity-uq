import os
from typing import Callable

import numpy as np
import torch
from laplace import ParametricLaplace
from torch import nn

from helpers import timestamped_filename


# noinspection PyPep8Naming
class IO_Helper:
    def __init__(
            self,
            base_folder,
            arrays_folder="arrays",
            models_folder="models",
            plots_folder="plots",
            metrics_folder='metrics',
    ):
        self.arrays_folder = os.path.join(base_folder, arrays_folder)
        self.models_folder = os.path.join(base_folder, models_folder)
        self.plots_folder = os.path.join(base_folder, plots_folder)
        self.metrics_folder = os.path.join(base_folder, metrics_folder)
        self.folders = [self.arrays_folder, self.models_folder, self.plots_folder, self.metrics_folder]
        for folder in self.folders:
            os.makedirs(folder, exist_ok=True)

    ### GETTERS ###
    def get_array_savepath(self, filename):
        return os.path.join(self.arrays_folder, filename)

    def get_model_savepath(self, filename):
        return os.path.join(self.models_folder, filename)

    def get_plot_savepath(self, filename):
        return os.path.join(self.plots_folder, filename)

    def get_metrics_savepath(self, filename):
        return os.path.join(self.metrics_folder, filename)

    ### LOADERS ###
    def load_array(self, filename):
        return np.load(self.get_array_savepath(filename))

    def load_model(self, filename):
        import pickle
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
        filepath = self.get_model_savepath(filename)
        kwargs['weights_only'] = False
        model = torch.load(filepath, *args, **kwargs)
        model.eval()
        return model

    def load_torch_model_statedict(self, model_class, filename, **kwargs):
        path = self.get_model_savepath(filename)
        state_dict = torch.load(path)
        model = model_class(**kwargs)
        model.load_state_dict(state_dict)
        return model

    def load_laplace_model_statedict(
            self,
            base_model: nn.Module,
            la_instantiator: Callable[[nn.Module], ParametricLaplace],
            laplace_model_filename: str,
    ):
        laplace_model_filepath = self.get_model_savepath(laplace_model_filename)
        la = la_instantiator(base_model)
        la.load_state_dict(torch.load(laplace_model_filepath))
        return la

    ### SAVERS ###
    def save_array(self, array, filename):
        path = self.get_array_savepath(filename)
        np.save(path, array)

    def save_model(self, model, filename):
        import pickle
        path = self.get_model_savepath(filename)
        with open(path, "wb") as file:
            pickle.dump(model, file)

    def save_torch_model(self, model, filename):
        path = self.get_model_savepath(filename)
        torch.save(model, path)

    def save_torch_model_statedict(self, model, filename):
        path = self.get_model_savepath(filename)
        torch.save(model.state_dict(), path)

    def save_laplace_model_statedict(
            self,
            laplace_model,
            laplace_model_filename,
    ):
        laplace_model_filepath = self.get_model_savepath(laplace_model_filename)
        torch.save(laplace_model.state_dict(), laplace_model_filepath)

    def save_plot(self, plotname=None):
        from matplotlib import pyplot as plt

        ext = os.path.splitext(plotname)[-1]
        if ext not in {'png', 'jpeg', 'jpg'}:
            print(f'filename {plotname} had no extension. saving as PNG')
            plotname += '.png'

        path = self.get_plot_savepath(plotname)
        plt.savefig(path)

    def save_metrics(self, metrics: dict, filename='metrics', add_timestamp=True):
        import json

        if add_timestamp:
            filename = timestamped_filename(filename, 'json')
        else:
            filename = f'{filename}.json'

        path = self.get_metrics_savepath(filename)
        metrics_str = json.dumps(metrics, indent=4)
        with open(path, 'w') as file:
            file.write(metrics_str)
