# based on: https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

import numpy as np
from multiprocessing import RawArray

from concurrent.futures import ProcessPoolExecutor


train_data = {}


def store_data_global(X_train_raw: RawArray, X_shape: tuple[int, int], y_train_raw: RawArray, y_shape: tuple[int, int]):
    train_data['X_train'] = X_train_raw
    train_data['X_shape'] = X_shape
    train_data['y_train'] = y_train_raw
    train_data['y_shape'] = y_shape


class Model:
    def __init__(self, n_models):
        self.model_ids = range(n_models)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        print('fitting...')
        print('making raw arrays')
        X_train_raw = self.get_raw_array(X_train)
        X_shape = X_train.shape
        y_train_raw = self.get_raw_array(y_train)
        y_shape = y_train.shape

        initargs = (X_train_raw, X_shape, y_train_raw, y_shape)
        print('submitting exec')
        with ProcessPoolExecutor(initializer=store_data_global, initargs=initargs) as executor:
            futures = [executor.submit(self.fit_single_model, model_id=model_id)
                       for model_id in self.model_ids]

        print('getting future results...')
        results = [future.result() for future in futures]
        print(f'results: {results}')

    @staticmethod
    def fit_single_model(model_id):
        print(f'{model_id}: loading data')
        X_np = np.frombuffer(train_data['X_train']).astype(np.float32).reshape(train_data['X_shape'])
        y_np = np.frombuffer(train_data['y_train']).astype(np.float32).reshape(train_data['y_shape'])
        print(f'{model_id}: computing result')
        result = np.sum(X_np) * np.sum(y_np) + model_id
        return result

    @staticmethod
    def get_raw_array(arr):
        assert len(arr.shape) == 2
        arr_raw = RawArray('d', arr.shape[0] * arr.shape[1])
        arr_raw_np = np.frombuffer(arr_raw).reshape(arr.shape)  # dont output this!
        np.copyto(arr_raw_np, arr)
        return arr_raw


def main():
    n_models = 5
    n_samples = 1000
    n_dim = 16

    X_shape = (n_samples, n_dim)
    X_train = np.random.randn(*X_shape)
    y_train = np.sin(X_train)

    model = Model(n_models)
    model.fit(X_train, y_train)


if __name__ == '__main__':
    main()
