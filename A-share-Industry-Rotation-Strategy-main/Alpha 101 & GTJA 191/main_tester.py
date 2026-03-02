import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# 从您的文件中导入Alphas类
from Alpha101_code_1 import Alphas, get_alpha

# 忽略一些在因子计算中可能出现的警告
warnings.filterwarnings("ignore")


class Config:
    # 请在这里填入您的Tushare Token
    TUSHARE_TOKEN = 'bb54b5627b861ff1bddb883cf0e820610bcd4fec05d347d40c60774c'
    
    # 回测时间范围
    START_DATE = '20210101'
    END_DATE = '20231231'
    
    # 股票池：这里以沪深300成分股为例，减少计算量。设为None则为全市场。
    # 首次运行时，可以先用少量股票测试，如：['000001.SZ', '600519.SH']
    STOCK_POOL = ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH'] # ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH'] 
    
    # 要测试的Alpha因子列表，None代表测试所有
    # 例如：['alpha001', 'alpha002', 'alpha003']
    ALPHAS_TO_TEST = ['alpha001']


def get_tushare_data(config):
    """从Tushare获取A股日线数据"""
    print("正在初始化Tushare...")
    pro = ts.pro_api(config.TUSHARE_TOKEN)
    
    print(f"正在获取股票池从 {config.START_DATE} 到 {config.END_DATE}...")
    if config.STOCK_POOL is None:
        all_stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
        stock_list = all_stocks['ts_code'].tolist()
    else:
        stock_list = config.STOCK_POOL

    all_data = []
    # 使用tqdm显示进度条
    for code in tqdm(stock_list, desc="下载日线数据"):
        try:
            df = ts.pro_bar(ts_code=code, start_date=config.START_DATE, end_date=config.END_DATE, adj='qfq')
            all_data.append(df)
        except Exception as e:
            print(f"获取 {code} 数据失败: {e}")
            
    if not all_data:
        raise ValueError("未能获取任何股票数据，请检查Token和网络。")

    # 合并所有数据
    full_df = pd.concat(all_data, ignore_index=True)
    
    # 数据预处理
    full_df['trade_date'] = pd.to_datetime(full_df['trade_date'])
    full_df = full_df.sort_values(by=['ts_code', 'trade_date']).reset_index(drop=True)
    
    print("数据下载和初步处理完成！")
    return full_df


def calculate_all_factors(data_df):
    """
    为数据计算所有101 Alpha因子
    """
    print("开始计算Alpha因子，这可能需要很长时间...")
    
    # 1. 字段名映射：将tushare的字段名改为alpha101需要的格式
    # S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_PCTCHANGE
    rename_dict = {
        'open': 'S_DQ_OPEN',
        'high': 'S_DQ_HIGH',
        'low': 'S_DQ_LOW',
        'close': 'S_DQ_CLOSE',
        'vol': 'S_DQ_VOLUME',
        'amount': 'S_DQ_AMOUNT',
        'pct_chg': 'S_DQ_PCTCHANGE'
    }
    data_df = data_df.rename(columns=rename_dict)
    
    # 2. 按股票分组，逐个计算因子
    # groupby().apply() 是一个非常强大的功能，但数据量大时会很慢
    factor_results = data_df.groupby('ts_code').apply(
        lambda x: get_alpha(x.set_index('trade_date'))
    )
    
    # 3. 将计算结果合并回原始DataFrame
    # factor_results 的索引是 (ts_code, trade_date)，需要重置
    factor_results = factor_results.reset_index()
    
    # 将因子数据与原始数据合并
    # 注意：原始的data_df没有被修改，我们用它的一个副本来合并
    final_df = pd.merge(data_df, factor_results, on=['ts_code', 'trade_date'], how='left')
    
    print("所有Alpha因子计算完成！")
    return final_df


class FactorAnalyzer:
    """
    一个简化的因子分析器，用于计算IC、分层收益和夏普比率
    """
    def __init__(self, factor_data, factor_col, return_col='next_return', n_groups=5):
        self.data = factor_data.copy()
        self.factor_col = factor_col
        self.return_col = return_col
        self.n_groups = n_groups
        self.cleaned_data = self._prepare_data()

    def _prepare_data(self):
        """数据清洗，去除NaN"""
        return self.data.dropna(subset=[self.factor_col, self.return_col])

    def calculate_ic(self):
        """计算信息系数 (Information Coefficient)"""
        # 按时间分组，计算每个截面上的因子值和未来收益的Spearman相关系数
        ic_series = self.cleaned_data.groupby('trade_date').apply(
            lambda x: x[self.factor_col].corr(x[self.return_col], method='spearman')
        )
        return ic_series.mean(), ic_series.std()

    def calculate_group_returns(self):
        """计算分层收益和多空组合收益"""
        # 1. 生成排名因子
        self.cleaned_data['rank'] = self.cleaned_data.groupby('trade_date')[self.factor_col].rank(method='first')
        
        # 2. 根据排名进行分组
        self.cleaned_data['group'] = self.cleaned_data.groupby('trade_date')['rank'].transform(
            lambda x: pd.qcut(x, self.n_groups, labels=False, duplicates='drop') + 1
        )
        
        # 3. 计算每组的平均收益率
        group_returns = self.cleaned_data.groupby(['trade_date', 'group'])[self.return_col].mean().unstack()
        
        # 4. 计算多空组合 (Top group - Bottom group)
        long_short_returns = group_returns[self.n_groups] - group_returns[1]
        
        return group_returns, long_short_returns

    def run_analysis(self):
        """运行全部分析"""
        if self.cleaned_data.empty:
            print(f"因子 {self.factor_col} 数据不足，跳过分析。")
            return
            
        # 计算IC
        ic_mean, ic_std = self.calculate_ic()
        
        # 计算分层收益
        group_returns, long_short_returns = self.calculate_group_returns()
        
        # 计算夏普比率 (假设无风险利率为0，年化)
        sharpe_ratio = (long_short_returns.mean() / long_short_returns.std()) * np.sqrt(252)
        
        # 打印结果
        print(f"\n--- 因子: {self.factor_col} 分析结果 ---")
        print(f"IC均值: {ic_mean:.4f}")
        print(f"IC标准差: {ic_std:.4f}")
        print(f"ICIR (IC均值/IC标准差): {ic_mean/ic_std:.4f}")
        print(f"多空组合年化夏普比率: {sharpe_ratio:.4f}")
        
        # 绘制分层累计收益图
        cumulative_group_returns = (1 + group_returns.fillna(0)).cumprod()
        cumulative_group_returns.plot(figsize=(12, 6), grid=True)
        plt.title(f'Factor "{self.factor_col}" Group Cumulative Returns')
        plt.ylabel('Cumulative Return')
        plt.xlabel('Date')
        plt.legend([f'Group {i}' for i in range(1, self.n_groups + 1)])
        plt.show()


if __name__ == '__main__':
    # 获取配置
    config = Config()
    
    # 步骤一：获取并处理数据
    try:
        market_data = get_tushare_data(config)
    except Exception as e:
        print(f"程序终止：{e}")
        exit()

    # 步骤二：计算所有因子
    factor_data = calculate_all_factors(market_data)
    
    # 步骤三：准备用于分析的数据 (计算未来收益)
    print("正在计算未来收益用于回测...")
    # 计算未来1天的收益率作为回测目标
    factor_data['next_return'] = factor_data.groupby('ts_code')['S_DQ_CLOSE'].transform(
        lambda x: x.pct_change().shift(-1)
    )
    
    # 步骤四：循环分析每个因子
    if config.ALPHAS_TO_TEST:
        alphas_to_run = config.ALPHAS_TO_TEST
    else:
        # 找出所有alphaXXX列
        alphas_to_run = [col for col in factor_data.columns if col.startswith('alpha')]

    for alpha_name in alphas_to_run:
        analyzer = FactorAnalyzer(factor_data, factor_col=alpha_name, n_groups=5)
        analyzer.run_analysis()

    print("\n所有因子分析完成！")