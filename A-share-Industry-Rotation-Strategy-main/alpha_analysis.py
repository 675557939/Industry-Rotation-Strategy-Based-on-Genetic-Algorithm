# -*- coding: utf-8 -*-
"""
截面因子分析框架（Pandas版）
Spearman IC / 分层回测 / 自动多空方向 / 9宫格分析图。
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False


class FactorAnalysis:

    def __init__(self, data, factor_col='MACD', return_col='next_return',
                 time_col='Time', universe=None, min_group_size=10):
        df = data.copy()
        if universe is not None and 'Tick' in df.columns:
            df = df[df['Tick'].isin(universe)]

        df = df.dropna(subset=[factor_col, return_col, time_col]).copy()
        df[time_col] = pd.to_datetime(df[time_col])
        for c in [factor_col, return_col]:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype('float32')

        self.data = df
        self.factor_col = factor_col
        self.return_col = return_col
        self.time_col = time_col
        self.min_group_size = min_group_size

    @staticmethod
    def _annualization_factor(index):
        if len(index) < 3:
            return 365.0
        idx = pd.DatetimeIndex(index)
        delta = np.median(np.diff(idx.values).astype('timedelta64[s]').astype('int64'))
        if delta <= 0:
            return 365.0
        return 365 * 24 * 3600 / float(delta)

    def calculate_ic(self):
        df = self.data.copy()
        df = df.assign(
            _rx=df.groupby(self.time_col)[self.factor_col].rank(method='average'),
            _ry=df.groupby(self.time_col)[self.return_col].rank(method='average'),
        )
        cnt = df.groupby(self.time_col)['_rx'].size()
        valid_times = cnt[cnt >= self.min_group_size].index
        df = df[df[self.time_col].isin(valid_times)]

        df = df.assign(
            _rx2=df['_rx'] ** 2, _ry2=df['_ry'] ** 2, _rxy=df['_rx'] * df['_ry'],
        )
        g = df.groupby(self.time_col, sort=True)
        agg = g[['_rx', '_ry', '_rx2', '_ry2', '_rxy']].sum()
        n = g.size().rename('n')
        agg = agg.join(n)

        cov_num = agg['_rxy'] - agg['_rx'] * agg['_ry'] / agg['n']
        var_x = agg['_rx2'] - agg['_rx'] ** 2 / agg['n']
        var_y = agg['_ry2'] - agg['_ry'] ** 2 / agg['n']
        ic = (cov_num / np.sqrt(var_x * var_y)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ic.name = 'IC'

        r = ic.values.astype('float64')
        nn = agg['n'].values.astype('float64')
        t_stat = r * np.sqrt(np.maximum(nn - 2, 1) / np.maximum(1 - r * r, 1e-12))
        p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=np.maximum(nn - 2, 1))

        return pd.DataFrame({'IC': ic.values.astype('float32'), 'p_value': p_value.astype('float32')},
                            index=ic.index)

    def factor_grouping(self, n_groups=5):
        df = self.data.copy()
        pct = df.groupby(self.time_col)[self.factor_col].rank(method='first', pct=True)
        grp = np.minimum((pct * n_groups).astype('int64'), n_groups - 1)

        tmp = df[[self.time_col, self.return_col]].copy()
        tmp['_g'] = grp.values
        res = (tmp.groupby([self.time_col, '_g'])[self.return_col]
               .mean().unstack('_g').sort_index()
               .rename(columns=lambda k: f'Group_{int(k)+1}'))
        return res.fillna(0.0).astype('float32')

    def run_full_analysis(self, n_groups=5):
        print('=' * 50)
        print(f'因子分析: {self.factor_col}')
        print('=' * 50)

        ic_df = self.calculate_ic()
        ic_mean = float(ic_df['IC'].mean())
        ic_std = float(ic_df['IC'].std(ddof=1))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0

        print(f'\n平均IC: {ic_mean:.4f} | IC标准差: {ic_std:.4f} | ICIR: {icir:.4f}')
        print(f'IC胜率: {(ic_df["IC"] > 0).mean():.2%} | 显著IC胜率(p<0.05): {(ic_df["p_value"] < 0.05).mean():.2%}')

        direction = 'LS' if ic_mean > 0 else ('SL' if ic_mean < 0 else 'NONE')
        print(f'多空方向: {direction}')

        group_returns = self.factor_grouping(n_groups)
        cumulative_returns = (1.0 + group_returns).cumprod()
        ANN = self._annualization_factor(group_returns.index)

        if not group_returns.empty:
            print('\n分组收益:')
            for col in group_returns.columns:
                r = group_returns[col].astype('float64')
                ann_ret = r.mean() * ANN
                vol = r.std(ddof=1) * np.sqrt(ANN)
                sharpe = ann_ret / vol if vol > 0 else 0.0
                print(f'  {col}: 年化={ann_ret:.2%}, Sharpe={sharpe:.2f}')

        long_short, cumulative_ls = None, None
        if f'Group_{n_groups}' in group_returns.columns and 'Group_1' in group_returns.columns:
            if direction == 'LS':
                long_short = group_returns[f'Group_{n_groups}'] - group_returns['Group_1']
            elif direction == 'SL':
                long_short = group_returns['Group_1'] - group_returns[f'Group_{n_groups}']
            else:
                long_short = group_returns[f'Group_{n_groups}'] - group_returns['Group_1']

            cumulative_ls = (1.0 + long_short).cumprod()
            r = long_short.astype('float64')
            ann_ret = r.mean() * ANN
            vol = r.std(ddof=1) * np.sqrt(ANN)
            sharpe = ann_ret / vol if vol > 0 else 0.0
            print(f'\n多空({direction}): 年化={ann_ret:.2%}, Sharpe={sharpe:.2f}')

        ic_df = ic_df.copy()
        ic_df['Year'] = ic_df.index.year
        ic_df['Month'] = ic_df.index.month
        monthly_ic = ic_df.groupby(['Year', 'Month'])['IC'].mean().reset_index()
        monthly_ic['Date'] = pd.to_datetime(monthly_ic[['Year', 'Month']].assign(day=1))

        mic_mean = float(monthly_ic['IC'].mean())
        mic_std = float(monthly_ic['IC'].std(ddof=1))
        mic_ir = mic_mean / mic_std if mic_std > 0 else 0.0
        print(f'\n月度IC: {mic_mean:.4f} | 月度ICIR: {mic_ir:.4f}')

        return ic_df, group_returns, cumulative_returns, monthly_ic, cumulative_ls

    def plot_analysis(self, ic_df, group_returns, cumulative_returns, monthly_ic, cumulative_ls=None):
        fig = plt.figure(figsize=(20, 16))

        ax = plt.subplot(3, 3, 1)
        ax.plot(ic_df.index, ic_df['IC'], alpha=0.7)
        ax.plot(ic_df.index, ic_df['IC'].rolling(30).mean(), linewidth=2, label='30D MA')
        ax.axhline(y=0, linestyle='--', alpha=0.5)
        ax.set_title('IC Time Series', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = plt.subplot(3, 3, 2)
        ax.hist(ic_df['IC'], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=ic_df['IC'].mean(), linestyle='--', label=f"Mean: {ic_df['IC'].mean():.4f}")
        ax.set_title('IC Distribution', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = plt.subplot(3, 3, 3)
        for col in cumulative_returns.columns:
            ax.plot(cumulative_returns.index, cumulative_returns[col], label=col, linewidth=2)
        ax.set_title('Cumulative Returns', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

        if cumulative_ls is not None:
            ax = plt.subplot(3, 3, 4)
            ax.plot(cumulative_ls.index, cumulative_ls, linewidth=3)
            ax.set_title('Long-Short Returns', fontweight='bold'); ax.grid(True, alpha=0.3)

        ax = plt.subplot(3, 3, 5)
        ax.plot(monthly_ic['Date'], monthly_ic['IC'], marker='o', linewidth=2, markersize=4)
        ax.axhline(y=0, linestyle='--', alpha=0.5)
        ax.set_title('Monthly IC', fontweight='bold'); ax.grid(True, alpha=0.3)

        ax = plt.subplot(3, 3, 6)
        ax.plot(ic_df.index, ic_df['IC'].cumsum(), linewidth=2)
        ax.set_title('Cumulative IC', fontweight='bold'); ax.grid(True, alpha=0.3)

        ax = plt.subplot(3, 3, 7)
        g = group_returns.dropna()
        if not g.empty:
            ax.boxplot([g[c].dropna() for c in g.columns], labels=g.columns)
        ax.set_title('Returns by Quintile', fontweight='bold'); ax.grid(True, alpha=0.3)

        ax = plt.subplot(3, 3, 8)
        roll = ic_df['IC'].rolling(60)
        mean60 = roll.mean()
        std60 = roll.std()
        ax.plot(ic_df.index, mean60, label='60D Mean', linewidth=2)
        ax.fill_between(ic_df.index, mean60 - std60, mean60 + std60, alpha=0.3, label='+/-1std')
        ax.set_title('Rolling IC', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = plt.subplot(3, 3, 9)
        yearly_ic = ic_df.groupby(ic_df.index.year)['IC'].mean()
        ax.bar(yearly_ic.index.astype(str), yearly_ic.values, alpha=0.7)
        ax.set_title('Annual Average IC', fontweight='bold'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
