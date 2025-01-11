from typing import Callable, Literal, TYPE_CHECKING
import logging
import os

import numpy as np

if TYPE_CHECKING:
    import torch
    import laplace


# noinspection PyPep8Naming
class IO_Helper:
    def __init__(
            self,
            base_folder,
            methods_kwargs=None,
            filename_parts: dict[tuple[list[tuple[str, str]], str]] = None,  # todo: make optional?
            filesave_prefix=None,
            arrays_folder="arrays",
            models_folder="models",
            plots_folder="plots",
            loss_plots_folder='plots_losses',
            metrics_folder='metrics',
            sep='_',
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
        :param methods_kwargs: dict of (method_name: method_kwargs_dict) pairs
        :param arrays_folder:
        :param models_folder:
        :param plots_folder:
        :param metrics_folder:
        :param sep:
        :param filesave_prefix:
        """
        warn_msg = 'IO_Helper param {} is None, so not all functionality will be available!'
        if filename_parts is None:
            logging.warning(warn_msg.format('filename_parts'))
        if methods_kwargs is None:
            logging.warning(warn_msg.format('methods_kwargs'))
        self.filename_parts = filename_parts
        self.methods_kwargs = methods_kwargs
        self.filesave_prefix = filesave_prefix
        self.sep = sep
        self.arrays_folder = os.path.join(base_folder, arrays_folder)
        self.models_folder = os.path.join(base_folder, models_folder)
        self.plots_folder = os.path.join(base_folder, plots_folder)
        self.loss_plots_folder = os.path.join(base_folder, loss_plots_folder)
        self.metrics_folder = os.path.join(base_folder, metrics_folder)
        self.folders = [self.arrays_folder, self.models_folder, self.plots_folder, self.metrics_folder]
        for folder in self.folders:
            os.makedirs(folder, exist_ok=True)

    ### GETTERS ###
    def _get_array_savepath(self, filename):
        return os.path.join(self.arrays_folder, filename)

    def _get_model_savepath(self, filename):
        return os.path.join(self.models_folder, filename)

    def _get_plot_savepath(self, filename, is_loss_plot=False):
        plots_folder = self.loss_plots_folder if is_loss_plot else self.plots_folder
        return os.path.join(plots_folder, filename)

    def _get_metrics_savepath(self, filename):
        return os.path.join(self.metrics_folder, filename)

    ### LOADERS ###
    def load_array(self, method_name=None, filename=None, infix=None):
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='array')
        return np.load(self._get_array_savepath(filename))

    def load_model(self, method_name=None, filename=None, infix=None):
        import pickle
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='model')
        with open(self._get_model_savepath(filename), "rb") as file:
            model = pickle.load(file)
        return model

    def load_torch_model(self, method_name=None, filename=None, infix=None):
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='model')
        import torch
        torch.set_default_dtype(torch.float32)
        filepath = self._get_model_savepath(filename)
        model = torch.load(filepath, weights_only=False)
        model.eval()
        return model

    def load_torch_model_statedict(self, model_class, method_name=None, filename=None, model_kwargs=None, infix=None):
        """

        :param infix:
        :param method_name:
        :param model_class: constructor for the model class. Must be able to accept all arguments as keyword arguments.
        :param model_kwargs: kwargs passed to the model_class constructor
        :param filename:
        :return:
        """
        import torch
        torch.set_default_dtype(torch.float32)
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='model')
        if model_kwargs is None:
            model_kwargs = {}
        path = self._get_model_savepath(filename)
        state_dict = torch.load(path)
        model = model_class(**model_kwargs)
        model.load_state_dict(state_dict)
        return model

    def load_laplace_model_statedict(
            self,
            base_model: 'torch.nn.Module',
            la_instantiator: Callable[['torch.nn.Module'], 'laplace.ParametricLaplace'],
            method_name=None,
            filename: str = None,
            infix=None,
    ):
        import torch
        torch.set_default_dtype(torch.float32)
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='model')
        laplace_model_filepath = self._get_model_savepath(filename)
        la = la_instantiator(base_model)
        la.load_state_dict(torch.load(laplace_model_filepath))
        return la

    ### SAVERS ###
    def save_array(self, array, method_name=None, filename=None, infix=None):
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='array')
        path = self._get_array_savepath(filename)
        np.save(path, array)

    def save_model(self, model, method_name=None, filename=None, infix=None):
        import pickle
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='model')
        path = self._get_model_savepath(filename)
        logging.info(f'saving model in path {path}')
        with open(path, "wb") as file:
            # noinspection PyTypeChecker
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    def save_torch_model(self, model, method_name=None, filename=None, infix=None):
        import torch
        torch.set_default_dtype(torch.float32)
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='model')
        path = self._get_model_savepath(filename)
        torch.save(model, path)

    def save_torch_model_statedict(self, model, method_name=None, filename=None, infix=None):
        import torch
        torch.set_default_dtype(torch.float32)
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='model')
        path = self._get_model_savepath(filename)
        torch.save(model.state_dict(), path)

    def save_laplace_model_statedict(
            self,
            laplace_model,
            method_name=None,
            filename=None,
            infix=None,
    ):
        import torch
        torch.set_default_dtype(torch.float32)
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='model')
        laplace_model_filepath = self._get_model_savepath(filename)
        torch.save(laplace_model.state_dict(), laplace_model_filepath)

    def save_plot(self, method_name=None, filename=None, infix=None, is_loss_plot=False):
        """

        :param is_loss_plot: only relevant if file_type='plot' determines where to store the plot.
        :param method_name:
        :param filename:
        :param infix:
        :return:
        """
        from matplotlib import pyplot as plt
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='plot')
        else:
            ext = os.path.splitext(filename)[-1]
            if ext not in {'png', 'jpeg', 'jpg'}:
                logging.info(f"filename '{filename}' had no extension. saving as PNG")
                filename += '.png'
        path = self._get_plot_savepath(filename, is_loss_plot=is_loss_plot)
        plt.savefig(path)

    def save_metrics(self, metrics: dict, method_name=None, filename=None, infix=None):
        import json
        infix = 'metrics' if infix is None else f'{infix}_metrics'
        if filename is None:
            filename = self.make_filename(method_name, infix=infix, file_type='metrics')
        elif not os.path.splitext(filename)[-1]:
            filename += '.json'
        path = self._get_metrics_savepath(filename)
        metrics_str = json.dumps(metrics, indent=4)
        with open(path, 'w') as file:
            file.write(metrics_str)

    def make_filename(
            self,
            method_name,
            infix=None,
            file_type: Literal['model', 'plot', 'array', 'metrics'] = 'model',
    ):
        """
        make *model* filename by default (with corresp. ending)

        :param method_name: name of the method calling this function. must be one of the
         'native_...', 'posthoc_...' or 'base_...' methods
        :param infix:
        :param file_type: one of 'model', 'plot', 'array', 'metrics'
        :return:
        """
        # todo: docstring
        # todo: better handling!
        method_name_parts = method_name.split(2 * self.sep)  # take care of posthoc_model__base_model naming
        if len(method_name_parts) == 2:
            method_name, base_suffix = method_name_parts
        else:
            method_name = method_name_parts[0]
            base_suffix = None
        kwargs = self.methods_kwargs[method_name]
        suffixes, model_ext = self.filename_parts[method_name]

        match file_type:
            case 'model':
                ext = model_ext
            case 'plot':
                ext = 'png'
            case 'array':
                ext = 'npy'
            case 'metrics':
                ext = 'json'
            case _:
                raise ValueError(f'filetype must be one of model, plot, array, metrics. received: {file_type}')
        joined_suffixes = [self.filesave_prefix] if self.filesave_prefix is not None else []
        for shorthand, kwarg_name in suffixes:
            joined_suffix = f'{shorthand}{kwargs[kwarg_name]}'
            joined_suffixes.append(joined_suffix)
        suffix_str = self.sep.join(joined_suffixes)

        filename = f'{method_name}_{base_suffix}' if base_suffix is not None else method_name
        if infix is not None:
            filename += f'{self.sep}{infix}'
        filename += f'{self.sep}{suffix_str}.{ext}'
        return filename
