import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(n_points_temp, filepath='data.pkl', input_cols=None, output_cols=None):
    """load and prepare data"""
    if input_cols is None:
        input_cols = [
            'load_last_week',
            'load_last_hour',
            'load_now',
            'is_workday',
            'is_saturday_and_not_holiday',
            'is_sunday_or_holiday',
            'is_heating_period',
        ]
    if output_cols is None:
        output_cols = ['load_next_hour']
    df = pd.read_pickle(filepath)

    mid = df.shape[0] // 2
    X = df[input_cols].iloc[mid - n_points_temp: mid + n_points_temp]
    y = df[output_cols].iloc[mid - n_points_temp: mid + n_points_temp]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)
    return X_train, X_test, y_train, y_test

