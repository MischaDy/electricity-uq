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


def update_training_flags():
    logging.info('updating training flags...')
    assert not (settings.DO_TRAIN_ALL and settings.SKIP_TRAINING_ALL)

    for _, method_kwargs in settings.METHODS_KWARGS.items():
        if settings.DO_TRAIN_ALL:
            method_kwargs['skip_training'] = False
        elif settings.SKIP_TRAINING_ALL:
            method_kwargs['skip_training'] = True


def update_run_size_setup():
    run_size_settings = RUN_SIZE_DICT.get(settings.RUN_SIZE)
    if run_size_settings is None:
        return

    for setting_name, setting_value in run_size_settings.items():
        setattr(settings, setting_name, setting_value)


def update_progress_bar_settings():
    if settings.SHOW_PROGRESS_BARS is not None:
        for _, method_kwargs in settings.METHODS_KWARGS.items():
            if 'show_progress_bar' not in method_kwargs:
                continue
            method_kwargs['show_progress_bar'] = settings.SHOW_PROGRESS_BARS


def update_losses_plots_settings():
    if settings.SHOW_LOSSES_PLOTS is not None:
        for _, method_kwargs in settings.METHODS_KWARGS.items():
            if 'show_losses_plot' not in method_kwargs:
                continue
            method_kwargs['show_losses_plot'] = settings.SHOW_LOSSES_PLOTS
    if settings.SAVE_LOSSES_PLOTS is not None:
        for _, method_kwargs in settings.METHODS_KWARGS.items():
            if 'save_losses_plot' not in method_kwargs:
                continue
            method_kwargs['save_losses_plot'] = settings.SAVE_LOSSES_PLOTS
