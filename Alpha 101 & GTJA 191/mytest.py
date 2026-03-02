import pandas as pd
import numpy as np
from Alpha_code_1 import get_alpha
from pathlib import Path

file_path = Path('CI指数_日线数据.csv')
index_codes = [
    'CI005001.INDX', 'CI005002.INDX', 'CI005008.INDX', 'CI005014.INDX',
    'CI005019.INDX', 'CI005021.INDX', 'CI005024.INDX', 'CI005028.INDX',
    'CI005029.INDX', 'CI005030.INDX'
]

all_alphas = []

df = pd.read_csv(file_path)
df['amount'] = df['close'] * df['volume']
df['outstanding_share'] = 1e8
df['turnover'] = df['volume'] / df['outstanding_share']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['order_book_id', 'date']).reset_index(drop=True)

for code in index_codes:
    stock_df = df[df['order_book_id'] == code].copy()
    if stock_df.empty:
        continue
    stock_df = stock_df.set_index('date')
    stock_df['change'] = stock_df['close'].pct_change()
    stock_df.dropna(subset=['change'], inplace=True)
    try:
        alpha = get_alpha(stock_df)
        alpha['idx'] = code
        alpha['date'] = alpha.index
        all_alphas.append(alpha)
    except Exception as e:
        print(f"计算 {code} Alpha失败: {e}")

if all_alphas:
    final_df = pd.concat(all_alphas, ignore_index=False).reset_index()
    final_df.to_csv('alpha101.csv', index=False)
    print(f"Alpha因子已保存至 alpha101.csv")
