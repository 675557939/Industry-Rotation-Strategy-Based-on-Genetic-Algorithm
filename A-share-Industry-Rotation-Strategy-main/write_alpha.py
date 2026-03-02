import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def load_data(start_date, end_date):
    df = pd.read_csv('CI指数_日线数据.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df.rename(columns={'order_book_id': 'idx', 'date': 'Time'}, inplace=True)
    df = df.sort_values(by=['idx', 'Time'])

    alpha = df[['idx', 'Time']].copy()
    alpha['next_return'] = df.groupby('idx')['close'].transform(lambda x: x.shift(-1) / x - 1)
    return df, alpha
