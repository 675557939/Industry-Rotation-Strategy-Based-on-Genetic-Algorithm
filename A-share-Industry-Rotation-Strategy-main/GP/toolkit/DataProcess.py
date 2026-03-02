import numpy as np
import pandas as pd
import dill


def load_timing_data():
    train_data = pd.read_csv('./data/IC_train.csv', index_col=0)
    test_data = pd.read_csv('./data/IC_test.csv', index_col=0)
    train_data.index = pd.to_datetime(train_data.index)
    test_data.index = pd.to_datetime(test_data.index)
    feature_names = list(train_data.columns)
    train_data['y'] = np.log(train_data['Open'].shift(-4) / train_data['Open'].shift(-1))
    train_data.dropna(inplace=True)
    return train_data, test_data, feature_names


def load_selecting_data():
    vwap = pd.read_csv('../data/daily_5/vwap.csv', index_col=[0], parse_dates=[0])
    close = pd.read_csv('../data/daily_5/close.csv', index_col=[0], parse_dates=[0])
    buy_vol = pd.read_csv('../data/daily_5/buy_volume_exlarge_order.csv', index_col=[0], parse_dates=[0])
    y_train1 = np.log(vwap.shift(-5) / vwap.shift(-1))
    y_train2 = np.log(vwap.shift(-10) / vwap.shift(-1))
    x_dict = {'close': close, 'buy_volume_exlarge_order': buy_vol}
    x_array = np.transpose(np.array(list(x_dict.values())), axes=(1, 2, 0))
    feature_names = list(x_dict.keys())
    return x_array, feature_names


def process_factor_data():
    with open('./data/stock_data.pickle', 'rb') as f:
        price = dill.load(f)
    price = price.loc['2015':, :]
    price.dropna(how='all', axis=1, inplace=True)

    with open('./data/stock_data.pickle', 'rb') as f:
        price = dill.load(f)
    with open('./data/factor_data.pickle', 'rb') as f:
        x_dict = dill.load(f)

    for key in list(x_dict.keys())[:-1]:
        x_dict[key] = x_dict[key].loc[price.index, price.columns]
    with open('./data/factor_data.pickle', 'wb') as f:
        dill.dump(x_dict, f)

    new_dict = {key: x_dict[key].loc[price.index, price.columns]
                for key in list(x_dict.keys())[:-1]}
    with open('./data/what.pickle', 'wb') as f:
        dill.dump(new_dict, f)
    return x_dict


def delete_factor_data(path):
    data = pd.read_csv(path, index_col=[0], parse_dates=[0])
    data = data.loc['2010':, :]
    data.to_csv(path)
