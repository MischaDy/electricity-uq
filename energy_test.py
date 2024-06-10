#!/usr/bin/env python
# coding: utf-8


import numpy as np



def get_data(as_df: bool):
    # # Prepare Data
        
    import pandas as pd
    
    pd.options.mode.copy_on_write = True
    
    YEARS = [2023]
    TZ = 'Europe/Berlin'
    
    df = pd.read_csv(f'data_Energy_Germany/Realised_Demand_Germany_{YEARS[0]}.csv', sep=';')
    
    df.drop(['Residual Load [MWh]', 'Pumped Storage [MWh]', 'Date to'], axis=1, inplace=True)
    df.rename(columns={'Date from': 'date_from', 'Total (Grid Load) [MWh]': 'grid_load'}, inplace=True)
    
    # convert to datetime columns
    
    df['date_from'] = pd.to_datetime(df['date_from'], format='%d.%m.%y %H:%M').dt.tz_localize(TZ, ambiguous='infer')
    
    # convert to float column
    df.grid_load = df.grid_load.str.replace('.', '').str.replace(',', '.')  # clean float representation
    df = df.astype({'grid_load': float})
    
    for col in df.columns:
        if any(df.date_from.isna()):
            print(f'WARNING: Missing value in column {col} detected!')
    
    
    
    from holidays.utils import country_holidays
    
    
    # === categorize days ===
    
    german_holidays = country_holidays('DE', years=YEARS)
    holiday_dates = set(german_holidays.keys())
    
    def is_sunday_or_holiday(timestamp):
        return timestamp.date() in holiday_dates or timestamp.day_name() == 'Sunday'
    
    def gen_saturday_col(df):
        return df.date_from.map(lambda ts: ts.day_name() == 'Saturday') & ~df.is_sunday_or_holiday
    
    df = df.assign(is_sunday_or_holiday=df.date_from.map(is_sunday_or_holiday))
    df = df.assign(is_saturday_and_not_holiday=gen_saturday_col(df))
    df = df.assign(is_workday=(~df.is_sunday_or_holiday & ~df.is_saturday_and_not_holiday))
    
    
    
    # === add heating period ===
    
    # heating period is approx. from October to March, i.e. not April (4) to September (9)
    df = df.assign(is_heating_period=df.date_from.map(lambda ts: ts.month not in range(4, 9+1)))
    
    
    
    grid_load_dict = dict(zip(df.date_from, df.grid_load))
    
    
    
    from datetime import timedelta
    
    # === add lagged values for input and output ===
    
    
    def add_one_week(ts):
        return add_timedelta(ts, timedelta(weeks=1))
    
    def sub_one_year(ts):
        # todo: decide how to handle leap years
        return add_timedelta(ts, timedelta(days=-365))
    
    def add_timedelta(ts, td):
        # based on: https://www.hacksoft.io/blog/handling-timezone-and-dst-changes-with-python
        ts_utc = ts.astimezone('UTC')
        ts_next_week_utc = ts_utc + td
        ts_next_week = ts_next_week_utc.astimezone(TZ)
        dst_offset_diff = ts.dst() - ts_next_week.dst()
        ts_next_week += dst_offset_diff
        return ts_next_week
    
    
    df_points_per_hour = 4
    
    df.rename(columns={'date_from': 'ts_last_week', 'grid_load': 'load_last_week'}, inplace=True)
    
    df = df.assign(ts_now=df.ts_last_week.map(add_one_week))
    
    # we ignore DST changes for 1h lags, but account for them for lags >= 1d
    df = df.assign(ts_last_hour=df.ts_now.shift(df_points_per_hour))
    df = df.assign(ts_next_hour=df.ts_now.shift(-df_points_per_hour))
    df.dropna(inplace=True, ignore_index=True)
    df.head()
    
    
    
    # === remove timestamps for which no grid load data is available
    
    ts_with_data_min, ts_with_data_max = df.ts_last_week.min(), df.ts_last_week.max()
    
    for col in ['ts_now', 'ts_last_hour', 'ts_next_hour']:
        out_of_bounds_index = df[(df[col] < ts_with_data_min) | (df[col] > ts_with_data_max)].index
        df.drop(index=out_of_bounds_index, inplace=True)
    
    
    
    # === add grid load columns by time stamp ===
    
    df = df.assign(load_last_hour=df.ts_last_hour.map(lambda ts: grid_load_dict[ts]))
    df = df.assign(load_now=df.ts_now.map(lambda ts: grid_load_dict[ts]))
    df = df.assign(load_next_hour=df.ts_next_hour.map(lambda ts: grid_load_dict[ts]))
    
    # remove unneeded time stamp columns
    df.drop(['ts_last_week', 'ts_last_hour', 'ts_now'], axis=1, inplace=True)


def df_to_input_output(df, with_ts=False):
    input_cols = [
        'load_last_week',
        'load_last_hour',
        'load_now',
        'is_workday',
        'is_saturday_and_not_holiday',
        'is_sunday_or_holiday',
        'is_heating_period',
    ]
    output_cols = ['load_next_hour']
    ts_cols = ['ts_next_hour']
    
    X = np.array(df[input_cols], dtype=float).reshape(-1, len(input_cols))  # (n_samples, n_features)
    y = np.array(df[output_cols], dtype=float).reshape(-1, len(output_cols))  # (n_samples, n_targets)
    ts = np.array(df[ts_cols]).reshape(-1, len(ts_cols))
    
    return (X, y, ts) if with_ts else (X, y)