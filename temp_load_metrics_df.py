import pandas as pd

FILENAME = 'metrics_comparison_test.csv'


def print_df(df):
    # 'display.max_rows', None,
    with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
        print(df)


def get_metrics_df(filename):
    df = pd.read_csv(
        filename,
        sep=';',
        header=1,
        # index_col=0,
    )
    df.rename(columns={'Unnamed: 0': 'metrics'}, inplace=True)
    df.drop(columns=['optimal', 'miserable'], inplace=True)
    df.drop(index=df[df.metrics == 'SMAPE'].index, inplace=True)
    return df


df = get_metrics_df(FILENAME)
print_df(df)
