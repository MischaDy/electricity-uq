import logging
import settings


RUN_SIZE_DICT = {
    'small': {
        'DATA_FILEPATH': 'data/data_1600.pkl',
        'N_POINTS_PER_GROUP': 800,
        'TRAIN_YEARS': None,
        'VAL_YEARS': None,
        'TEST_YEARS': None,
    },
    'big': {
        'DATA_FILEPATH': 'data/data_2015_2018.pkl',
        'N_POINTS_PER_GROUP': None,
        'TRAIN_YEARS': (2016, 2017),
        'VAL_YEARS': (2017, 2018),
        'TEST_YEARS': (2018, 2019),
    },
    'full': {
        'DATA_FILEPATH': 'data/data.pkl',
        'N_POINTS_PER_GROUP': None,
        'TRAIN_YEARS': (2016, 2022),
        'VAL_YEARS': (2022, 2023),
        'TEST_YEARS': (2023, 2024),
    },
}


def update_training_flags(do_train_all=None, skip_training_all=None):
    if do_train_all is None:
        do_train_all = settings.DO_TRAIN_ALL
    if skip_training_all is None:
        skip_training_all = settings.SKIP_TRAINING_ALL
    assert not (do_train_all and skip_training_all)  # can legitimately be None!

    for _, method_kwargs in settings.METHODS_KWARGS.items():
        if do_train_all:
            method_kwargs['skip_training'] = False
        elif skip_training_all:
            method_kwargs['skip_training'] = True


def update_run_size_setup(run_size=None):
    if run_size is None:
        run_size = settings.RUN_SIZE
    run_size_settings = RUN_SIZE_DICT.get(run_size)
    if run_size_settings is None:
        return

    for setting_name, setting_value in run_size_settings.items():
        setattr(settings, setting_name, setting_value)


def update_progress_bar_settings(show_progress_bars=None):
    if show_progress_bars is None:
        show_progress_bars = settings.SHOW_PROGRESS_BARS
    if show_progress_bars is not None:
        for _, method_kwargs in settings.METHODS_KWARGS.items():
            if 'show_progress_bar' not in method_kwargs:
                continue
            method_kwargs['show_progress_bar'] = settings.SHOW_PROGRESS_BARS


def update_losses_plots_settings(show_losses_plots=None, save_losses_plots=None):
    if show_losses_plots is None:
        show_losses_plots = settings.SHOW_LOSSES_PLOTS
    if save_losses_plots is None:
        save_losses_plots = settings.SAVE_LOSSES_PLOTS
    if show_losses_plots is not None:
        for _, method_kwargs in settings.METHODS_KWARGS.items():
            if 'show_losses_plot' not in method_kwargs:
                continue
            method_kwargs['show_losses_plot'] = settings.SHOW_LOSSES_PLOTS
    if save_losses_plots is not None:
        for _, method_kwargs in settings.METHODS_KWARGS.items():
            if 'save_losses_plot' not in method_kwargs:
                continue
            method_kwargs['save_losses_plot'] = settings.SAVE_LOSSES_PLOTS
