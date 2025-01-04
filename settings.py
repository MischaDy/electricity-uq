import logging

from scipy import stats

### CONVENIENCE FLAGS ###

DO_BIG_RUN = True
DO_SMALL_RUN = False

DO_TRAIN_ALL = False
SKIP_TRAINING_ALL = False


### NORMAL SETTINGS ###

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

DATA_FILEPATH = 'data/data_1600.pkl'  # 'data/data_2015_2018.pkl'
N_POINTS_PER_GROUP = 800  # None
TRAIN_YEARS = None  # (2016, 2017)  # todo: simplify
VAL_YEARS = None  # (2017, 2018)  # todo: simplify
TEST_YEARS = None  # (2018, 2019)  # todo: simplify

STANDARDIZE_DATA = True

PLOT_DATA = False
PLOT_UQ_RESULTS = True
PLOT_BASE_RESULTS = True
SHOW_PLOTS = False
SAVE_PLOTS = True

SKIP_BASE_MODEL_COPY = True
SHOULD_SAVE_RESULTS = True
USE_FILESAVE_PREFIX = True

LOGGING_LEVEL = logging.INFO

STORAGE_PATH = "comparison_storage"

METHOD_WHITELIST = [
    'base_model_linreg',
    # 'base_model_nn',
    # 'base_model_rf',
    # 'native_gpytorch',
    # 'native_mvnn',
    # 'native_quantile_regression_nn',
    'posthoc_conformal_prediction',
    # 'posthoc_laplace_approximation',
]

METHODS_KWARGS = {
    "native_mvnn": {
        'skip_training': False,
        "n_iter": 100,
        "num_hidden_layers": 2,
        "hidden_layer_size": 50,
        "activation": None,  # defaults to leaky ReLU
        "lr": 1e-4,
        "lr_patience": 30,
        "regularization": 0,  # 1e-2,
        "warmup_period": 50,
        "frozen_var_value": 0.1,
        'show_losses_plot': False,
        'save_losses_plot': True,
        'save_model': True,
    },
    "native_quantile_regression_nn": {
        'skip_training': False,
        'n_iter': 100,
        'num_hidden_layers': 2,
        'hidden_layer_size': 50,
        'activation': None,
        'random_seed': 42,
        'lr': 1e-4,
        'use_scheduler': True,
        'lr_patience': 30,
        'regularization': 0,
        'show_progress': False,
        'show_losses_plot': False,
        'save_losses_plot': True,
        'save_model': True,
    },
    "native_gpytorch": {
        'skip_training': False,
        'n_iter': 100,
        'lr': 1e-2,
        'use_scheduler': True,
        'lr_patience': 30,
        'lr_reduction_factor': 0.5,
        'show_progress': False,
        'show_plots': True,
        'show_losses_plot': False,
        'save_losses_plot': True,
        'save_model': True,
    },
    "posthoc_conformal_prediction": {
        "skip_training": False,
        "n_estimators": 2,
        "verbose": 1,
        "save_model": True,
    },
    "posthoc_laplace_approximation": {
        'skip_training': False,
        "n_iter": 100,
        'save_model': True,
    },
    "base_model_linreg": {
        "skip_training": False,
        "n_jobs": -1,
        "save_model": True,
    },
    "base_model_rf": {
        "skip_training": True,
        'model_param_distributions': {
            "max_depth": stats.randint(2, 50),
            "n_estimators": stats.randint(10, 200),
        },
        'cv_n_iter': 10,
        'cv_n_splits': 5,
        "random_seed": 42,
        "verbose": 4,
        'n_jobs': -1,
        "save_model": True,
    },
    "base_model_nn": {
        "skip_training": True,
        "n_iter": 100,
        'num_hidden_layers': 2,
        'hidden_layer_size': 50,
        'activation': None,
        "lr": 1e-2,
        "lr_patience": 30,
        "lr_reduction_factor": 0.5,
        "show_progress_bar": True,
        "show_losses_plot": True,
        "save_losses_plot": True,
        "random_seed": 42,
        "verbose": 1,
        "save_model": True,
    },
}

FILENAME_PARTS = {
    "native_mvnn": (
        [
            ('it', 'n_iter'),
            ('nh', 'num_hidden_layers'),
            ('hs', 'hidden_layer_size'),
        ],
        'pth'
    ),
    "native_quantile_regression_nn": (
        [
            ('it', 'n_iter'),
            ('nh', 'num_hidden_layers'),
            ('hs', 'hidden_layer_size'),
        ],
        'pth'
    ),
    "native_gpytorch": (
        [
            ('it', 'n_iter'),
        ],
        'pth'
    ),
    "posthoc_conformal_prediction": (
        [
            ('it', 'n_estimators'),
        ],
        'model'
    ),
    "posthoc_laplace_approximation": (
        [
            ('it', 'n_iter'),
        ],
        'pth'
    ),
    "base_model_linreg": (
        [
        ],
        'model'
    ),
    "base_model_nn": (
        [
            ('it', 'n_iter'),
            ('nh', 'num_hidden_layers'),
            ('hs', 'hidden_layer_size'),
        ],
        'pth'
    ),
    "base_model_rf": (
        [
            ('it', 'cv_n_iter'),
            ('its', 'cv_n_splits'),
        ],
        'model'
    ),
}
