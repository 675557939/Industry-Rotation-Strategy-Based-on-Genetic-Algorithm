import pandas as pd
import numpy as np
import statsmodels.api as sm
import akshare as ak
from Alpha_code_1 import *

# 设置基本参数
start_date = '2016-02-01'  # 回测起始日期
end_date = '2024-12-31'   # 回测结束日期

# 获取股票历史数据
stock_df = ak.stock_zh_a_daily(symbol='sh600588', start_date=start_date, end_date=end_date, adjust='qfq')
df = stock_df.set_index('date')
df['change'] = df['close'].pct_change()
# 删除第一行并重新赋予序号
df = df.iloc[1:].reset_index(drop=True)
alpha = get_alpha(df)
print('finish!')
print(alpha.tail())