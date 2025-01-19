import logging
from typing import Iterable, TYPE_CHECKING, Generator

from helpers.io_helper import IO_Helper

if TYPE_CHECKING:
    import numpy as np


def get_method_to_arrs_gen(
        method_to_arr_names_dict: dict[str, Iterable[str]],
        methods_whitelist: set[str] = None,
        io_helper=None,
        storage_path='comparison_storage',
) -> Generator[tuple[str, 'np.ndarray'], None, None]:
    for method, arr_names in method_to_arr_names_dict.items():
        if methods_whitelist is not None and method not in methods_whitelist:
            continue
        try:
            arrs = load_arrs(arr_names, io_helper=io_helper, storage_path=storage_path)
        except FileNotFoundError as e:
            logging.error(f"when loading arrays for {method}, file '{e.filename}' couldn't be found."
                          f" skipping method.")
            continue
        yield method, arrs


def load_arrs(filenames: Iterable, io_helper=None, storage_path='comparison_storage'):
    if io_helper is None:
        io_helper = IO_Helper(storage_path)
    return [io_helper.load_array(filename=filename) for filename in filenames]
