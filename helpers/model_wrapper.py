from helpers import misc_helpers


class ModelWrapper:
    def __init__(self, model, output_dim: int = 2):
        """

        :param model:
        :param output_dim: must be either 1 or 2
        """
        assert output_dim in {1, 2}
        self.model_ = model
        self.output_dim = output_dim
        self.output_dim_orig = output_dim

    def predict(self, input_):
        output = self.model_.predict(input_)
        output = misc_helpers.make_arr_1d(output) if self.output_dim == 1 else misc_helpers.make_arr_2d(output)
        return output

    def set_output_dim(self, output_dim, orig=False):
        self.output_dim = output_dim
        if orig:
            self.output_dim_orig = output_dim

    def reset_output_dim(self):
        self.output_dim = self.output_dim_orig

    def __call__(self, input_):
        return self.predict(input_)

    def __getattr__(self, item):
        model = self.__getattribute__('model_')  # workaround bcdirect attr access doesn't work
        return getattr(model, item)

    def __setattr__(self, key, value):
        self.__dict__[key] = value
