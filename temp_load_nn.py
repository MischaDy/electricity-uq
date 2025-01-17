import torch.nn

from helpers import misc_helpers
from helpers.io_helper import IO_Helper
from src_base_models.nn_estimator import NN_Estimator


X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers._quick_load_data('full')
dim_in = X_train.shape[1]
model_kwargs = {
    'dim_in': dim_in,
    'train_size_orig': 210432,
    "n_iter": 2,
    "num_hidden_layers": 2,
    "hidden_layer_size": 50,
    'activation': torch.nn.LeakyReLU,  # defaults to leaky ReLU
    "weight_decay": 1e-3,
    "lr": 1e-5,  # defaults to 1e-2 if use_scheduler is true
    'use_scheduler': True,
    "lr_patience": 10,
    "lr_reduction_factor": 0.8,
    "show_progress_bar": True,
    "show_losses_plot": False,
    "save_losses_plot": True,
    'n_samples_train_loss_plot': 10000,
    "random_seed": 42,
    "verbose": 1,
    'early_stop_patience': 30,
}
io_helper = IO_Helper()
model = io_helper.load_torch_model_statedict(model_class=NN_Estimator, model_kwargs=model_kwargs,
                                             filename='base_model_nn_n210432_it300_nh2_hs50.pth')
# io_helper.save_torch_model(model, filename='base_model_nn_n210432_it200_nh2_hs50_model2.pth')
