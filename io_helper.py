import os

import numpy as np
import torch


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
        import pickle
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
        filepath = self.get_model_savepath(filename)
        model = model_class(*args, **kwargs)
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
        import pickle
        path = self.get_model_savepath(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(model, file)

    def save_torch_model(self, model, filename):
        path = self.get_model_savepath(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)

    def save_plot(self, filename):
        from matplotlib import pyplot as plt

        path = self.get_plot_savepath(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
