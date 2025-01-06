import logging

from scipy import stats

### CONVENIENCE FLAGS ###

DO_BIG_RUN = False
DO_SMALL_RUN = False

DO_TRAIN_ALL = True
SKIP_TRAINING_ALL = False


### NORMAL SETTINGS ###

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

DATA_FILEPATH = 'data/data.pkl'
N_POINTS_PER_GROUP = None
TRAIN_YEARS = (2016, 2022)  # todo: simplify
VAL_YEARS = (2022, 2023)  # todo: simplify
TEST_YEARS = (2023, 2024)  # todo: simplify

STANDARDIZE_DATA = True

SHOW_PLOTS = False
SAVE_PLOTS = True
PLOT_DATA = False
PLOT_BASE_RESULTS = True
PLOT_UQ_RESULTS = True
PLOT_BASE_RESULTS_PARTIAL = True

SHOW_PROGRESS_BARS = False  # bool or None
SHOW_LOSSES_PLOTS = False  # bool or None
SAVE_LOSSES_PLOTS = True  # bool or None

SKIP_BASE_MODEL_COPY = True
SHOULD_SAVE_RESULTS = True
USE_FILESAVE_PREFIX = True

LOGGING_LEVEL = logging.INFO

STORAGE_PATH = "comparison_storage"

METHOD_WHITELIST = [
    # 'base_model_linreg',
    # 'base_model_nn',
    # 'base_model_hgbr',
    'native_gpytorch',
    'native_mvnn',
    'native_quantile_regression_nn',
    # 'posthoc_conformal_prediction',
    # 'posthoc_laplace_approximation',
]

METHODS_KWARGS = {
    "native_mvnn": {
        'skip_training': False,
        "n_iter": 150,
        "num_hidden_layers": 2,
        "hidden_layer_size": 50,
        "activation": None,  # defaults to leaky ReLU
        "lr": 1e-4,
        "lr_patience": 30,
        "weight_decay": 1e-3,
        "warmup_period": 50,
        "frozen_var_value": 0.1,
        'show_losses_plot': False,
        'save_losses_plot': True,
        'show_progress_bar': False,
        'save_model': True,
    },
    "native_quantile_regression_nn": {
        'skip_training': False,
        "n_iter": 150,
        "num_hidden_layers": 2,
        "hidden_layer_size": 50,
        'activation': None,
        'random_seed': 42,
        'lr': 1e-4,
        'use_scheduler': True,
        'lr_patience': 30,
        "weight_decay": 1e-3,
        'show_progress_bar': False,
        'show_losses_plot': False,
        'save_losses_plot': True,
        'save_model': True,
    },
    "native_gpytorch": {
        'skip_training': False,
        'n_iter': 300,
        'lr': 1e-2,
        'use_scheduler': True,
        'lr_patience': 30,
        'lr_reduction_factor': 0.5,
        'show_progress_bar': False,
        'show_plots': True,
        'show_losses_plot': False,
        'save_losses_plot': True,
        'save_model': True,
    },
    "posthoc_conformal_prediction": {
        "skip_training": False,
        "n_estimators": 10,
        "verbose": 1,
        "save_model": True,
    },
    "posthoc_laplace_approximation": {
        'skip_training': False,
        "n_iter": 300,
        'save_model': True,
        'verbose': True,
        'show_progress_bar': False,
    },
    "base_model_linreg": {
        "skip_training": True,
        "n_jobs": -1,
        "save_model": True,
    },
    "base_model_hgbr": {
        "skip_training": True,
        'model_param_distributions': {
            # 'max_features': stats.randint(1, X_train.shape[1]),
            "max_iter": stats.randint(10, 1000),
            'learning_rate': stats.loguniform(0.015, 0.15),
            'max_leaf_nodes': stats.randint(10, 100),
            'min_samples_leaf': stats.randint(15, 100),
            'l2_regularization': [0, 1e-4, 1e-3, 1e-2, 1e-1],
        },
        'cv_n_iter': 50,
        'cv_n_splits': 5,
        "random_seed": 42,
        "verbose": 1,
        'n_jobs': -1,
        "save_model": True,
    },
    "base_model_nn": {
        "skip_training": True,
        "n_iter": 300,
        "num_hidden_layers": 3,
        "hidden_layer_size": 80,
        'activation': None,
        "weight_decay": 1e-3,
        "lr": 1e-2,
        "lr_patience": 30,
        "lr_reduction_factor": 0.5,
        "show_progress_bar": False,
        "show_losses_plot": False,
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
    "base_model_hgbr": (
        [
            ('it', 'cv_n_iter'),
            ('its', 'cv_n_splits'),
        ],
        'model'
    ),
}
