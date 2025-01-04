from typing import Literal

from helpers import misc_helpers


class ModelWrapper:
    def __init__(self, model, output_dim: Literal[1, 2] = 2):
        self.model_ = model
        self.output_dim = output_dim
        self._output_dim_old = None

    def predict(self, input_):
        output = self.model_.predict(input_)
        output = misc_helpers.make_arr_1d(output) if self.output_dim == 1 else misc_helpers.make_arr_2d(output)
        return output

    def set_output_dim(self, output_dim):
        self._output_dim_old = self.output_dim
        self.output_dim = output_dim

    def reset_output_dim(self):
        # switch values around
        self._output_dim_old, self.output_dim = self.output_dim, self._output_dim_old

    def __call__(self, input_):
        return self.predict(input_)

    def __getattr__(self, item):
        model = self.__getattribute__('model_')  # workaround bcdirect attr access doesn't work
        return getattr(model, item)

    def __setattr__(self, key, value):
        self.__dict__[key] = value
