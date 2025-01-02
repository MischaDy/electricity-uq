from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np


TrainValTestDataFrames = tuple['pd.DataFrame', 'pd.DataFrame', 'pd.DataFrame', 'pd.DataFrame', 'pd.DataFrame',
                               'pd.DataFrame']
UQ_Output = tuple[
    'np.ndarray',
    Union['np.ndarray', None],
    Union['np.ndarray', None]
]
