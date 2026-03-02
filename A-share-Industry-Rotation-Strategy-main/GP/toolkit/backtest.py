import os
import pickle
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tqdm
import itertools
import multiprocessing
import dill
from abc import abstractmethod, ABCMeta
from collections import abc
from itertools import zip_longest
from functools import wraps

logging.getLogger('matplotlib').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', message='FixedFormatter should only be used together with FixedLocator')
os.makedirs('./result/backtest/', exist_ok=True)

if os.name == 'posix':
    plt.rcParams['font.sans-serif'] = ['Songti SC']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class SiMuPaiPaiWang:
    """排排网配色方案"""
    colors = {'strategy': '#de3633', 'benchmark': '#80b3f6', 'excess': '#f4b63f'}

    def __getitem__(self, key):
        return self.colors[key]

    def __repr__(self):
        return self.colors.__repr__()


class Displayer:
    def __init__(self, df: pd.DataFrame):
        assert isinstance(df.index, pd.DatetimeIndex)
        for col in ['benchmark_curve', 'strategy_curve', 'position', 'signal']:
            assert col in df.columns
        self.df = df
        self.freq_base = self.df['benchmark'].resample('d').apply(lambda x: len(x))[0]
        self.holding_infos = self._calc_holding_infos()
        self.out_stats = self._calc_stats()

    def get_results(self):
        return self.out_stats, self.holding_infos, self.df

    def save_results(self, file_name=''):
        self.out_stats.to_csv(f'./result/backtest/{file_name}_stats.csv')
        self.holding_infos.to_csv(f'./result/backtest/{file_name}_holdings.csv')
        self.df.to_csv(f'./result/backtest/{file_name}_trading_details.csv')

    def _calc_holding_infos(self):
        state = self.df['position'].copy(deep=True)
        assert not all(state == 0), '没有进行交易'

        time_info = [num for num, count in enumerate(np.abs((state - state.shift(1)).fillna(0)))
                     for i in range(int(count))]
        open_time = time_info[::2]
        exit_time = time_info[1::2]
        holding_infos = pd.DataFrame(zip_longest(open_time, exit_time, fillvalue=None))
        holding_infos.columns = ['open_time', 'exit_time']
        holding_infos.fillna(len(self.df), inplace=True)
        holding_infos['direction'] = state[list(holding_infos['open_time'])].values
        holding_infos['holding_time'] = holding_infos['exit_time'] - holding_infos['open_time']
        holding_infos['returns'] = holding_infos.apply(lambda x:
            np.log(self.df['strategy_curve'].iloc[int(x['exit_time']) - 1] /
                   self.df['strategy_curve'].iloc[int(x['open_time']) - 1]), axis=1)
        holding_infos['open_time_stamp'] = self.df.index[holding_infos['open_time'].values - 1]
        holding_infos['exit_time_stamp'] = self.df.index[holding_infos['exit_time'].values.astype(int) - 1]
        return holding_infos

    def _calc_stats(self):
        output_stat = {}
        strategy_returns = np.log(self.df['strategy_curve'] / self.df['strategy_curve'].shift(1))
        benchmark_returns = np.log(self.df['benchmark_curve'] / self.df['benchmark_curve'].shift(1))
        excess_returns = strategy_returns - benchmark_returns

        output_stat['Annualized_Mean'] = 252 * strategy_returns.groupby(strategy_returns.index.date).sum().mean()
        output_stat['Annualized_Std'] = np.sqrt(252) * strategy_returns.groupby(strategy_returns.index.date).sum().std()
        output_stat['Sharpe'] = output_stat['Annualized_Mean'] / output_stat['Annualized_Std']
        output_stat['Excess_Annualized_Mean'] = 252 * self.freq_base * np.mean(excess_returns)
        output_stat['Excess_Annualized_Std'] = np.sqrt(252 * self.freq_base) * np.std(excess_returns)
        output_stat['Excess_sharpe'] = output_stat['Excess_Annualized_Mean'] / output_stat['Excess_Annualized_Std']
        output_stat['MaxDrawDown'] = ((self.df['strategy_curve'].cummax() - self.df['strategy_curve']) /
                                      self.df['strategy_curve'].cummax()).max()

        for direction, label in [(1, 'Long'), (-1, 'Short')]:
            try:
                vc = self.holding_infos['direction'].value_counts()
                output_stat[f'{label}Counts'] = vc[direction]
                grouped = self.holding_infos.groupby('direction')
                output_stat[f'Mean{label}Time'] = grouped['holding_time'].mean()[direction]
                output_stat[f'Per{label}Return'] = grouped['returns'].mean()[direction]
            except KeyError:
                output_stat[f'{label}Counts'] = 0
                output_stat[f'Mean{label}Time'] = 0
                output_stat[f'Per{label}Return'] = 0

        try:
            temp_p = self.holding_infos['returns'][self.holding_infos['returns'] > 0].mean()
            temp_n = self.holding_infos['returns'][self.holding_infos['returns'] < 0].mean()
            output_stat['PnL'] = np.abs(temp_p / temp_n)
        except ZeroDivisionError:
            output_stat['PnL'] = np.inf

        output_stat['WinRate'] = (self.holding_infos['returns'] > 0).sum() / len(self.holding_infos)
        return pd.Series(output_stat)

    def plot_(self, comm='', tick_count=9, plot_name='', plot_PnL=True, show_bool=False):
        datetime_index = self.df.index
        if plot_PnL:
            strategy_returns = self.df['strategy_curve']
            benchmark_returns = self.df['benchmark_curve']
            excess_returns = (1 + strategy_returns.pct_change() - benchmark_returns.pct_change()).cumprod()
            y_hlines = 1
        else:
            strategy_returns = self.df['strategy_curve'].pct_change().cumsum().fillna(0)
            benchmark_returns = self.df['benchmark_curve'].pct_change().cumsum().fillna(0)
            excess_returns = (strategy_returns - benchmark_returns)
            y_hlines = 0

        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(12, 1)
        ax1 = fig.add_subplot(gs[:8, :])
        ax2 = fig.add_subplot(gs[8:10, :])
        ax3 = fig.add_subplot(gs[10:, :])
        fontsize = 12

        plot_df1 = pd.concat([strategy_returns, benchmark_returns, excess_returns], axis=1)
        plot_df1.columns = ['strategy', 'benchmark', 'excess']
        x = range(len(self.df))
        for col, label in zip(plot_df1.columns, ['策略', '基准', '超额']):
            ax1.plot(x, plot_df1[col].values, label=label, color=SiMuPaiPaiWang()[col])
        ax1.hlines(y=y_hlines, xmin=0, xmax=len(self.df), color='grey', linestyles='dashed')

        step = max(int(len(self.df) / tick_count), 3)
        ax1.set_xlim(0, len(self.df))
        ax1.grid(True)
        ax1.set_xticks(range(len(self.df))[::step])
        ax1.set_xticklabels([''] * len(ax1.get_xticks()))
        ax1.set_title(f'{plot_name} 费率{comm}', fontsize=fontsize)
        ax1.legend()
        ax1.set_ylabel('累计份额', fontsize=fontsize)

        ax2.plot(range(len(self.df)), self.df.position, color='tab:grey')
        ax2.set_xlim(0, len(self.df))
        ax2.set_xticklabels([''] * len(ax1.get_xticks()))
        ax2.set_ylabel('仓位', fontsize=fontsize)

        strategy_curve = np.array(self.df.strategy_curve.ffill().fillna(1))
        max_dd = (np.maximum.accumulate(strategy_curve) - strategy_curve) / np.maximum.accumulate(strategy_curve)
        ax3.plot(range(len(self.df)), -max_dd, color='tab:purple')
        ax3.set_xlim(0, len(self.df))
        ax3.set_xticks(range(len(self.df))[::step])
        ytick_list = [-np.round(max(max_dd), 3), -np.round(max(max_dd) * 0.5, 3), 0]
        ax3.set_yticks(ytick_list)
        ax3.set_yticklabels([f'{v:.0%}' for v in ytick_list])
        ax3.set_xticklabels(datetime_index.strftime('%Y-%m-%d')[::step])
        ax3.set_ylabel('最大回撤', fontsize=fontsize)

        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        plt.tight_layout()
        plt.savefig(f'./result/backtest/{plot_name} {comm}.png', bbox_inches='tight', dpi=300)
        if show_bool:
            plt.show()
        return fig


class BackTester(object):
    """向量化回测框架，仅支持ALL-IN"""
    __metaclass__ = ABCMeta

    @staticmethod
    def process_strategy(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            self.create_backtest_env()
            if set(self.params).issubset(set(kwargs.keys())):
                for name in self.params:
                    self.params[name] = kwargs[name]
            else:
                for idx, name in enumerate(self.params):
                    self.params[name] = args[1:][idx]

            func(self, *args[1:], **kwargs)
            assert 'signal' in self.backtest_env.columns, '未计算信号'
            assert 'position' in self.backtest_env.columns, '未填充持仓'
            assert self.backtest_env['position'].isnull().sum() == 0, '无交易记录'

            self.backtest_env['benchmark'] = np.log(
                self.backtest_env['transact_base'] / self.backtest_env['transact_base'].shift(1))
            self.backtest_env['strategy'] = self.backtest_env['position'] * self.backtest_env['benchmark']
            self.backtest_env['benchmark_curve'] = self.backtest_env['benchmark'].cumsum().apply(np.exp)
            self.backtest_env['strategy_curve'] = self.backtest_env['strategy'].cumsum().apply(np.exp)

            if self.buy_commission is not None and self.sell_commission is not None:
                fees_factor = pd.Series(np.nan, index=self.backtest_env.index)
                pos_diff = self.backtest_env.position - self.backtest_env.position.shift(1)
                fees_factor[:] = np.where(pos_diff > 0, -pos_diff * self.buy_commission, np.nan)
                fees_factor[:] = np.where(pos_diff < 0, pos_diff * self.sell_commission, fees_factor)
                fees_factor.fillna(0, inplace=True)
                fees_factor += 1
                self.fees_factor = fees_factor
                self.backtest_env['strategy_curve'] *= fees_factor.cumprod()
                self.backtest_env['strategy'] = np.log(
                    self.backtest_env['strategy_curve'] / self.backtest_env['strategy_curve'].shift(1))
                self.backtest_env['strategy'].fillna(0, inplace=True)

            result = {}
            result['params'] = tuple(self.params.values())
            daily_ret = self.backtest_env['strategy'].groupby(self.backtest_env.index.date).sum()
            result['annualized_mean'] = 252 * daily_ret.mean()
            result['annualized_std'] = np.sqrt(252) * daily_ret.std()
            if result['annualized_std'] != 0:
                result['sharpe_ratio'] = result['annualized_mean'] / result['annualized_std']
            elif result['annualized_mean'] == 0:
                result['sharpe_ratio'] = 0
            elif result['annualized_mean'] < 0:
                result['sharpe_ratio'] = -999
            else:
                result['sharpe_ratio'] = 999
            cummax = np.maximum.accumulate(self.backtest_env['strategy_curve'].fillna(1))
            result['max_drawdown'] = np.max((cummax - self.backtest_env['strategy_curve']) / cummax)
            result['signal_counts'] = np.sum(np.abs(self.backtest_env['signal']))
            return result
        return wrapper

    def __init__(self, symbol_data: pd.DataFrame, transact_base='PreClose',
                 commissions=(0.23, 0.23), slippage_rate=None):
        assert isinstance(symbol_data, pd.DataFrame)
        assert isinstance(symbol_data.index, pd.DatetimeIndex)
        for attr in ['Open', 'High', 'Low', 'Close', 'Volume']:
            assert attr in symbol_data.columns

        self.data = symbol_data
        self.freq_base = self.data['Close'].resample('d').apply(lambda x: len(x))[0]
        self.transact_base = transact_base
        self.backtest_env = ...
        self.params = ...
        self.buy_commission = commissions[0]
        self.sell_commission = commissions[1]
        self.slippage_rate = slippage_rate
        self.init()

    def create_backtest_env(self):
        self.backtest_env = self.data.copy(deep=True)
        if self.transact_base == 'PreClose':
            self.backtest_env['transact_base'] = self.data['Close']
        elif self.transact_base == 'Open':
            self.backtest_env['transact_base'] = self.data['Open'].shift(-1).ffill()
        else:
            raise ValueError(f'transact_base must be "PreClose" or "Open", got {self.transact_base}')
        self.backtest_env['signal'] = np.nan
        self.backtest_env['position'] = np.nan

    @abstractmethod
    def init(self):
        self.params = ...
        raise NotImplementedError

    @property
    def params_name(self):
        try:
            return list(self.params.keys())
        except AttributeError:
            self.init()
            return list(self.params.keys())

    @abstractmethod
    @process_strategy.__get__(object)
    def run_(self, *args, **kwargs) -> dict:
        self.backtest_env.position = ...
        raise NotImplementedError

    def construct_position_(self, keep_raw=False, min_holding_period=None,
                            max_holding_period=None, take_profit=None, stop_loss=None):
        assert 'signal' in self.backtest_env.columns
        self.backtest_env['position'] = self.backtest_env['signal'].shift(1)

        if take_profit is not None and stop_loss is not None:
            mark = pd.Series(np.nan, index=self.backtest_env.index)
            mask = (self.backtest_env['position'] == 1) | (self.backtest_env['position'] == -1)
            mark[mask] = self.backtest_env['transact_base']
            mark.ffill(inplace=True)
            self.backtest_env['position'] = np.where(
                self.backtest_env['transact_base'] > mark * (1 + take_profit), 0, self.backtest_env['position'])
            self.backtest_env['position'] = np.where(
                self.backtest_env['transact_base'] < mark * (1 - stop_loss), 0, self.backtest_env['position'])

        if keep_raw:
            self.backtest_env['position'].fillna(0, inplace=True)
        else:
            if max_holding_period is not None:
                self.backtest_env['position'].ffill(limit=max_holding_period, inplace=True)
                self.backtest_env['position'].fillna(0, inplace=True)
            else:
                raise ValueError('max_holding_period should not be None if keep_raw is False')
        self.backtest_env.loc[self.backtest_env.index[0], 'position'] = 0

    def optimize_(self, goal='sharpe_ratio', method='grid', n_jobs=1, **kwargs):
        assert goal in ['annualized_mean', 'annualized_std', 'sharpe_ratio']
        for name in self.params:
            assert name in kwargs
            assert isinstance(kwargs[name], abc.Iterable)

        temp = itertools.product(*[kwargs[x] for x in self.params])
        if method == 'grid':
            if n_jobs > 1:
                with multiprocessing.Pool(n_jobs) as p:
                    results = p.starmap(self.run_, temp)
            else:
                results = [self.run_(*args) for args in temp]
            return max(results, key=lambda x: x[goal])

    def summary(self, *args, **kwargs) -> Displayer:
        return Displayer(self.backtest_env)

    def clear(self):
        del self.backtest_env

    @staticmethod
    def cross_up(series1, series2):
        return (series1 > series2) & (series1.shift(1) < series2.shift(1))

    @staticmethod
    def cross_down(series1, series2):
        return (series1 < series2) & (series1.shift(1) > series2.shift(1))
