# -*- coding: utf-8 -*-
"""
截面因子分析框架（Polars加速版）
支持 IC/ICIR 统计、分层回测、多空净值、费后净值。
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False


class FactorAnalysis:
    def __init__(self, data, factor_col='MACD', return_col='next_return',
                 time_col='Time', universe=None, min_group_size=10,
                 fee_bps=0.0, liquity_period=20, liquity_quantile=1.0,
                 min_listed_period=120):

        if isinstance(data, pd.DataFrame):
            self.lazy_df = pl.from_pandas(data).lazy()
        elif isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            self.lazy_df = data.lazy()
        else:
            raise ValueError("data 须为 pandas 或 polars DataFrame")

        self.factor_col = factor_col
        self.return_col = return_col
        self.time_col = time_col
        self.min_group_size = min_group_size
        self.fee_bps = float(fee_bps)
        self.liquity_period = int(liquity_period)
        self.liquity_quantile = float(liquity_quantile)
        self.min_listed_period = int(min_listed_period)

        if universe is not None:
            self.lazy_df = self.lazy_df.filter(pl.col('Tick').is_in(universe))

        self.lazy_df = (
            self.lazy_df
            .filter(
                pl.col(factor_col).is_not_null() &
                pl.col(return_col).is_not_null() &
                pl.col(time_col).is_not_null()
            )
            .with_columns([
                pl.col(time_col).cast(pl.Datetime),
                pl.col(factor_col).cast(pl.Float32),
                pl.col(return_col).cast(pl.Float32)
            ])
        )

        schema_names = set(self.lazy_df.collect_schema().names())
        has_tick = 'Tick' in schema_names

        # 上市期数过滤
        if has_tick and self.min_listed_period > 1:
            self.lazy_df = (
                self.lazy_df
                .sort(['Tick', self.time_col])
                .with_columns([
                    (pl.col(self.time_col).cum_count().over('Tick') + 1).alias('_listed_periods')
                ])
                .filter(pl.col('_listed_periods') >= self.min_listed_period)
            )

        # 流动性过滤
        if has_tick and self.liquity_quantile < 1.0:
            liq_col = next((c for c in ['Amount', 'Turnover', 'DollarVolume', 'Volume', 'volume']
                            if c in schema_names), None)
            q = max(min(self.liquity_quantile, 1.0), 0.0)
            if liq_col and self.liquity_period >= 1 and q > 0:
                self.lazy_df = (
                    self.lazy_df
                    .sort(['Tick', self.time_col])
                    .with_columns([
                        pl.col(liq_col).cast(pl.Float32)
                        .rolling_mean(window_size=self.liquity_period,
                                      min_periods=max(1, self.liquity_period // 2))
                        .over('Tick').alias('_liq_roll')
                    ])
                    .with_columns([
                        pl.col('_liq_roll').quantile(1.0 - q).over(self.time_col).alias('_liq_thr')
                    ])
                    .filter(pl.col('_liq_roll').is_not_null() & (pl.col('_liq_roll') >= pl.col('_liq_thr')))
                )

    @staticmethod
    def _annualization_factor_from_time_index(index):
        if index is None or len(index) < 3:
            return 252.0
        try:
            t = pd.DatetimeIndex(pd.to_datetime(index)).dropna()
            if len(t) < 3:
                return 252.0
            counts = pd.Series(1, index=t.normalize()).groupby(level=0).sum()
            if len(counts) == 0:
                return 252.0
            bars_per_day = float(counts.median())
            if not np.isfinite(bars_per_day) or bars_per_day <= 0:
                return 252.0
            return 252.0 * bars_per_day
        except Exception:
            return 252.0

    @staticmethod
    def _infer_freq_and_annualization(index):
        ann = FactorAnalysis._annualization_factor_from_time_index(index)
        if index is None or len(index) < 3:
            return ann, 'irregular', False
        try:
            t = pd.DatetimeIndex(pd.to_datetime(index)).dropna()
            if len(t) < 3:
                return ann, 'irregular', False
            counts = pd.Series(1, index=t.normalize()).groupby(level=0).sum()
            median_bars = float(counts.median()) if len(counts) else 1.0
            diffs = np.diff(t.view('int64') // 10**9)
            diffs = diffs[diffs > 0]
            if diffs.size == 0:
                return ann, 'irregular', False
            med = float(np.median(diffs))
            mad = float(np.median(np.abs(diffs - np.median(diffs))))
            is_regular = (mad / med) < 0.10
            if median_bars <= 1.5:
                label = 'daily' if is_regular else 'irregular'
            elif med >= 1800:
                label = 'hourly' if is_regular else 'irregular'
            else:
                label = 'minutely' if is_regular else 'irregular'
            return ann, label, is_regular
        except Exception:
            return ann, 'irregular', False

    @staticmethod
    def _annualization_factor(index):
        return FactorAnalysis._annualization_factor_from_time_index(index)

    def run_full_analysis(self, n_groups=5):
        print('=' * 60)
        print(f'因子分析: {self.factor_col}')
        print('=' * 60)

        # IC
        ic_agg = (
            self.lazy_df
            .group_by(self.time_col)
            .agg([
                pl.corr(self.factor_col, self.return_col, method='spearman').alias('IC'),
                pl.len().alias('n')
            ])
            .filter(pl.col('n') >= self.min_group_size)
            .sort(self.time_col)
        )

        # 分组收益
        group_agg = (
            self.lazy_df
            .with_columns([
                (
                    (pl.col(self.factor_col).rank(method='ordinal').over(self.time_col) /
                     pl.len().over(self.time_col)) * n_groups
                ).ceil().cast(pl.Int8).clip(1, n_groups).alias('group_id')
            ])
            .group_by([self.time_col, 'group_id'])
            .agg(pl.col(self.return_col).mean())
            .sort([self.time_col, 'group_id'])
        )

        ic_res_pl, group_res_pl = pl.collect_all([ic_agg, group_agg])

        # IC 统计
        ic_df = ic_res_pl.to_pandas().set_index(self.time_col).sort_index()
        ic_df['IC'] = ic_df['IC'].fillna(0.0).astype('float32')

        r = ic_df['IC'].values.astype('float64')
        n = ic_df['n'].values.astype('float64')
        t_stat = r * np.sqrt(np.maximum(n - 2, 1) / np.maximum(1 - r**2, 1e-12))
        ic_df['p_value'] = 2 * stats.t.sf(np.abs(t_stat), np.maximum(n - 2, 1))

        ic_mean = float(ic_df['IC'].mean())
        ic_std = float(ic_df['IC'].std(ddof=1))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0

        print(f'\n平均IC: {ic_mean:.4f} | IC标准差: {ic_std:.4f} | ICIR: {icir:.4f}')
        print(f'IC胜率: {(ic_df["IC"] > 0).mean():.2%}')

        direction = 'LS' if ic_mean < 0 else 'SL'
        if abs(ic_mean) < 1e-5:
            direction = 'NONE'
        print(f'多空方向: {direction}')

        # 分组
        group_df_long = group_res_pl.to_pandas()
        group_returns = group_df_long.pivot(index=self.time_col, columns='group_id', values=self.return_col)
        group_returns.columns = [f'Group_{int(c)}' for c in group_returns.columns]
        group_returns = group_returns.fillna(0.0).sort_index()

        cumulative_returns = (1.0 + group_returns).cumprod()

        ANN, freq_label, is_regular = self._infer_freq_and_annualization(group_returns.index)
        print(f'\n频率: {freq_label} | 年化系数: {ANN:.2f}')

        if not group_returns.empty:
            for col in group_returns.columns:
                r_arr = group_returns[col].values
                ann_ret = np.mean(r_arr) * ANN
                vol = np.std(r_arr, ddof=1) * np.sqrt(ANN)
                sharpe = ann_ret / vol if vol > 0 else 0.0
                print(f'{col}: 年化={ann_ret:.2%}, Sharpe={sharpe:.2f}')

        # 多空
        cumulative_ls = None
        g_first, g_last = 'Group_1', f'Group_{n_groups}'

        if g_last in group_returns.columns and g_first in group_returns.columns:
            if direction == 'LS':
                long_short = group_returns[g_first] - group_returns[g_last]
            else:
                long_short = group_returns[g_last] - group_returns[g_first]

            cumulative_ls_gross = (1.0 + long_short).cumprod()

            if self.fee_bps > 0:
                fee_per_period = 2.0 * (self.fee_bps / 1e4)
                ls_net_ret = (long_short - fee_per_period).clip(lower=-0.999999)
                cumulative_ls = pd.DataFrame(
                    {'Gross': cumulative_ls_gross, 'Net': (1.0 + ls_net_ret).cumprod()},
                    index=cumulative_ls_gross.index
                )
            else:
                cumulative_ls = cumulative_ls_gross

            r_ls = long_short.values
            ann_ret = np.mean(r_ls) * ANN
            vol = np.std(r_ls, ddof=1) * np.sqrt(ANN)
            sharpe = ann_ret / vol if vol > 0 else 0.0
            print(f'\n多空({direction}): 年化={ann_ret:.2%}, Sharpe={sharpe:.2f}')

        # 月度IC
        ic_df_m = ic_df.copy()
        ic_df_m['M'] = ic_df_m.index.to_period('M')
        monthly_ic = ic_df_m.groupby('M')['IC'].mean().reset_index()
        monthly_ic['Date'] = monthly_ic['M'].dt.to_timestamp()

        mic_mean = float(monthly_ic['IC'].mean())
        mic_std = float(monthly_ic['IC'].std(ddof=1))
        mic_ir = mic_mean / mic_std if mic_std > 0 else 0.0
        print(f'\n月度IC均值: {mic_mean:.4f} | 月度ICIR: {mic_ir:.4f}')

        return ic_df, group_returns, cumulative_returns, monthly_ic, cumulative_ls

    def plot_analysis(self, ic_df, group_returns, cumulative_returns, monthly_ic, cumulative_ls=None):
        fig = plt.figure(figsize=(20, 16))

        ax = plt.subplot(2, 2, 3)
        ax.hist(ic_df['IC'], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=ic_df['IC'].mean(), linestyle='--', label=f"Mean: {ic_df['IC'].mean():.4f}")
        ax.set_title('IC Distribution', fontsize=14, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = plt.subplot(2, 2, 1)
        for col in cumulative_returns.columns:
            ax.plot(cumulative_returns.index, cumulative_returns[col], label=col, linewidth=2)
        ax.set_yscale('log')
        ax.set_title('Cumulative Returns by Quintile (Log)', fontsize=14, fontweight='bold')
        ax.legend(); ax.grid(True, which='both', alpha=0.3)

        if cumulative_ls is not None:
            ax = plt.subplot(2, 2, 2)
            if isinstance(cumulative_ls, pd.DataFrame):
                for c in cumulative_ls.columns:
                    label = c if c == 'Gross' else f'{c} (fee={self.fee_bps:.1f}bps)'
                    ax.plot(cumulative_ls.index, cumulative_ls[c], linewidth=3, label=label)
            else:
                ax.plot(cumulative_ls.index, cumulative_ls, linewidth=3, label='Gross')
            ax.legend(); ax.set_yscale('log')
            ax.set_title('Long-Short Returns (Log)', fontsize=14, fontweight='bold')
            ax.grid(True, which='both', alpha=0.3)

        ax = plt.subplot(2, 2, 4)
        ax.plot(ic_df.index, ic_df['IC'].cumsum(), linewidth=2)
        ax.set_title('Cumulative IC', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
