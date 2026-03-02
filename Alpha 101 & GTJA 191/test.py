import pandas as pd
import akshare as ak
from Alpha_code_1 import get_alpha

start_date = '2016-02-01'
end_date = '2024-12-31'

stock_df = ak.stock_zh_a_daily(symbol='sh600588', start_date=start_date, end_date=end_date, adjust='qfq')
df = stock_df.set_index('date')
df['change'] = df['close'].pct_change()
df = df.iloc[1:].reset_index(drop=True)

alpha = get_alpha(df)
print(alpha.tail())
