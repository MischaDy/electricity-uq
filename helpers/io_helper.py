import logging
import os
from typing import Callable

import numpy as np


# noinspection PyPep8Naming
class IO_Helper:
    def __init__(
            self,
            base_folder,
            filename_parts: dict[tuple[list[tuple[str, str]], str]],  # todo: make optional?
            arrays_folder="arrays",
            models_folder="models",
            plots_folder="plots",
            metrics_folder='metrics',
    ):
        """

        :param base_folder:
        :param filename_parts: A dict that looks like this:
            {
                "method_name": ([('abbrev1', 'kwarg_name1'),
                                 ('abbrev2', 'kwarg_name2'),
                                 ], 'ext'),
                ...
            }
            and results in filenames such as: f'method_name_abbrev1{kwarg_name1}_abbrev2{kwarg_name2}.ext'.
        :param arrays_folder:
        :param models_folder:
        :param plots_folder:
        :param metrics_folder:
        """
        self.filename_parts = filename_parts
        self.arrays_folder = os.path.join(base_folder, arrays_folder)
        self.models_folder = os.path.join(base_folder, models_folder)
        self.plots_folder = os.path.join(base_folder, plots_folder)
        self.metrics_folder = os.path.join(base_folder, metrics_folder)
        self.folders = [self.arrays_folder, self.models_folder, self.plots_folder, self.metrics_folder]
        for folder in self.folders:
            os.makedirs(folder, exist_ok=True)

    ### GETTERS ###
    def get_array_savepath(self, method_name=None, method_kwargs=None, filename=None):
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        return os.path.join(self.arrays_folder, filename)

    def get_model_savepath(self, method_name=None, method_kwargs=None, filename=None):
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        return os.path.join(self.models_folder, filename)

    def get_plot_savepath(self, method_name=None, method_kwargs=None, filename=None):
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        return os.path.join(self.plots_folder, filename)

    def get_metrics_savepath(self, method_name=None, method_kwargs=None, filename=None):
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        return os.path.join(self.metrics_folder, filename)

    ### LOADERS ###
    def load_array(self, method_name=None, method_kwargs=None, filename=None):
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        return np.load(self.get_array_savepath(filename))

    def load_model(self, method_name=None, method_kwargs=None, filename=None):
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        import pickle
        with open(self.get_model_savepath(filename), "rb") as file:
            model = pickle.load(file)
        return model

    def load_torch_model(self, method_name=None, method_kwargs=None, filename=None):
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        import torch
        filepath = self.get_model_savepath(filename)
        model = torch.load(filepath, weights_only=False)
        model.eval()
        return model

    def load_torch_model_statedict(self, model_class, method_name=None, method_kwargs=None, filename=None, model_kwargs=None):
        """

        :param method_kwargs:
        :param method_name:
        :param model_class: constructor for the model class. Must be able to accept all arguments as keyword arguments.
        :param filename:
        :param model_kwargs: kwargs passed to the model_class constructor
        :return:
        """
        import torch

        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        path = self.get_model_savepath(filename)
        state_dict = torch.load(path)
        model = model_class(**model_kwargs)
        model.load_state_dict(state_dict)
        return model

    # noinspection PyUnresolvedReferences
    def load_laplace_model_statedict(
            self,
            base_model: 'torch.nn.Module',
            la_instantiator: Callable[['torch.nn.Module'], 'laplace.ParametricLaplace'],
            method_name=None,
            method_kwargs=None,
            filename: str = None,
    ):
        import torch
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        laplace_model_filepath = self.get_model_savepath(filename)
        la = la_instantiator(base_model)
        la.load_state_dict(torch.load(laplace_model_filepath))
        return la

    ### SAVERS ###
    def save_array(self, array, method_name=None, method_kwargs=None, filename=None):
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        path = self.get_array_savepath(filename)
        np.save(path, array)

    def save_model(self, model, method_name=None, method_kwargs=None, filename=None):
        import pickle
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        path = self.get_model_savepath(filename)
        with open(path, "wb") as file:
            # noinspection PyTypeChecker
            pickle.dump(model, file)

    def save_torch_model(self, model, method_name=None, method_kwargs=None, filename=None):
        import torch
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        path = self.get_model_savepath(filename)
        torch.save(model, path)

    def save_torch_model_statedict(self, model, method_name=None, method_kwargs=None, filename=None):
        import torch
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        path = self.get_model_savepath(filename)
        torch.save(model.state_dict(), path)

    def save_laplace_model_statedict(
            self,
            laplace_model,
            method_name=None,
            method_kwargs=None,
            filename=None
    ):
        import torch
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        laplace_model_filepath = self.get_model_savepath(filename)
        torch.save(laplace_model.state_dict(), laplace_model_filepath)

    def save_plot(self, method_name=None, method_kwargs=None, filename=None):
        from matplotlib import pyplot as plt
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        ext = os.path.splitext(filename)[-1]
        if ext not in {'png', 'jpeg', 'jpg'}:
            logging.info(f'filename {filename} had no extension. saving as PNG')
            filename += '.png'
        path = self.get_plot_savepath(filename)
        plt.savefig(path)

    def save_metrics(self, metrics: dict, method_name=None, method_kwargs=None, filename=None):
        import json
        if filename is None:
            filename = self.make_filename(method_name, method_kwargs)
        filename = f'{filename}_metrics'
        filename += '.json'
        path = self.get_metrics_savepath(filename)
        metrics_str = json.dumps(metrics, indent=4)
        with open(path, 'w') as file:
            file.write(metrics_str)

    def make_filename(self, method_name, kwargs, sep='_'):
        suffixes, ext = self.filename_parts[method_name]
        joined_suffixes = [f'{shorthand}{kwargs[kwarg_name]}'
                           for shorthand, kwarg_name in suffixes]
        suffix_str = sep.join(joined_suffixes)
        filename = f'{method_name}{sep}{suffix_str}.{ext}'
        return filename
