# -*- coding: utf-8 -*-
"""
行业轮动因子挖掘 - 滚动窗口GP训练
基于遗传编程在中信行业指数上挖掘截面因子，按月滚动训练并逐日生成因子值。
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit, prange
from toolkit.setupGPlearn import my_gplearn
from alpha_analysis import FactorAnalysis

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = './result/factor_validation/'
os.makedirs(RESULT_DIR, exist_ok=True)

FUNCTION_SET = ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                'abs', 'max', 'min', 'delayy', 'rankk']

FITNESS_MODE = 'mono_topret'


# Numba 加速 IC 计算

@jit(nopython=True)
def _rankdata_1d(arr):
    n = len(arr)
    ranks = np.empty(n, dtype=np.float64)
    order = np.argsort(arr)
    for i in range(n):
        ranks[order[i]] = float(i + 1)
    return ranks


@jit(nopython=True)
def _corr_1d(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    sum_x, sum_y = 0.0, 0.0
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
    mean_x = sum_x / n
    mean_y = sum_y / n

    cov, var_x, var_y = 0.0, 0.0, 0.0
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    if var_x < 1e-10 or var_y < 1e-10:
        return 0.0
    return cov / (np.sqrt(var_x) * np.sqrt(var_y))


@jit(nopython=True, parallel=True)
def _spearman_ic_2d(y_true, y_pred):
    """逐行计算 Spearman Rank IC"""
    n_days = y_true.shape[0]
    n_stocks = y_true.shape[1]
    ics = np.zeros(n_days, dtype=np.float64)

    for day in prange(n_days):
        y_row = y_true[day, :]
        pred_row = y_pred[day, :]

        valid_count = 0
        for j in range(n_stocks):
            if not (np.isnan(y_row[j]) or np.isnan(pred_row[j])):
                valid_count += 1
        if valid_count < 10:
            ics[day] = np.nan
            continue

        y_valid = np.empty(valid_count, dtype=np.float64)
        pred_valid = np.empty(valid_count, dtype=np.float64)
        idx = 0
        for j in range(n_stocks):
            if not (np.isnan(y_row[j]) or np.isnan(pred_row[j])):
                y_valid[idx] = y_row[j]
                pred_valid[idx] = pred_row[j]
                idx += 1

        ics[day] = _corr_1d(_rankdata_1d(y_valid), _rankdata_1d(pred_valid))
    return ics


# 冗余因子检测

def has_redundant_nesting(program_str):
    """检测一元函数冗余嵌套"""
    for func in ['log', 'sqrt', 'abs']:
        if f'{func}({func}(' in program_str:
            return True
    if re.search(r'(log|sqrt|abs)\(\s*(log|sqrt|abs)\(\s*(log|sqrt|abs)\(', program_str):
        return True
    if re.match(r'^(log|sqrt|abs)\([a-z_]+\)$', program_str.strip()):
        return True
    if program_str.strip() in ['open', 'high', 'low', 'close', 'volume', 'total_turnover']:
        return True
    return False


def count_binary_ops(program_str):
    binary_ops = ['add', 'sub', 'mul', 'div', 'max', 'min']
    return sum(program_str.count(op + '(') for op in binary_ops)


# 适应度函数

def score_func_ic(y, y_pred, sample_weight):
    if len(np.unique(y_pred[-1])) <= 10:
        return -999          # 退化因子给极低惩罚，确保任何有意义的因子都优于常数
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    pred_arr = np.ascontiguousarray(y_pred, dtype=np.float64)
    if FITNESS_MODE == 'mono_topret':
        return _fitness_mono_topret(y_arr, pred_arr)
    return _fitness_ic_winrate(y_arr, pred_arr)


def _calc_group_returns(y_true, y_pred, n_groups=5):
    n_days = y_true.shape[0]
    group_returns = np.zeros(n_groups, dtype=np.float64)
    group_counts = np.zeros(n_groups, dtype=np.float64)

    for day in range(n_days):
        y_row = y_true[day, :]
        pred_row = y_pred[day, :]
        valid_mask = ~(np.isnan(y_row) | np.isnan(pred_row))
        if np.sum(valid_mask) < n_groups * 3:
            continue

        y_valid = y_row[valid_mask]
        pred_valid = pred_row[valid_mask]
        sorted_idx = np.argsort(pred_valid)
        group_size = len(sorted_idx) // n_groups

        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else len(sorted_idx)
            group_returns[g] += np.mean(y_valid[sorted_idx[start:end]])
            group_counts[g] += 1

    if np.min(group_counts) < 1:
        return None
    return group_returns / (group_counts + 1e-10)


def _fitness_ic_winrate(y_true, y_pred, n_groups=5):
    """IC胜率 x |IC均值| + 分层约束 + 收益约束"""
    rank_ics = _spearman_ic_2d(y_true, y_pred)
    valid_ics = rank_ics[~np.isnan(rank_ics)]
    if len(valid_ics) < 5:
        return 0

    ic_mean = np.mean(valid_ics)
    win_rate = np.mean(np.array([1.0 if ic > 0 else 0.0 for ic in valid_ics]))
    if ic_mean < 0:
        win_rate = 1 - win_rate

    base_score = win_rate * abs(ic_mean) * 100

    avg_returns = _calc_group_returns(y_true, y_pred, n_groups)
    if avg_returns is None:
        return base_score * 0.3

    checks = [avg_returns[i] < avg_returns[i + 1] for i in range(n_groups - 1)]
    weights = [0.20, 0.15, 0.15, 0.20]
    penalties = [-0.40, -0.20, -0.15, -0.30]
    layer_bonus = sum(w if c else p for c, w, p in zip(checks, weights, penalties))

    top_ret_annual = avg_returns[-1] * 252
    spread_annual = (avg_returns[-1] - avg_returns[0]) * 252
    ret_bonus = 0.0
    ret_bonus += min(top_ret_annual * 0.5, 0.3) if top_ret_annual > 0 else -0.3
    ret_bonus += min(spread_annual * 0.3, 0.3) if spread_annual > 0 else -0.2

    score = np.clip(base_score * (1 + layer_bonus + ret_bonus), -10, 10)
    return score if not np.isnan(score) else 0


def _fitness_mono_topret(y_true, y_pred, n_groups=5):
    """单调性 x 多空价差（对熊市鲁棒）"""
    n_days = y_true.shape[0]
    group_returns = np.zeros(n_groups, dtype=np.float64)
    group_counts = np.zeros(n_groups, dtype=np.float64)
    spread_returns = []   # Top组 - Bottom组 的日度价差

    for day in range(n_days):
        y_row = y_true[day, :]
        pred_row = y_pred[day, :]
        valid_mask = ~(np.isnan(y_row) | np.isnan(pred_row))
        if np.sum(valid_mask) < n_groups * 3:
            continue

        y_valid = y_row[valid_mask]
        pred_valid = pred_row[valid_mask]
        sorted_idx = np.argsort(pred_valid)
        group_size = len(sorted_idx) // n_groups

        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else len(sorted_idx)
            group_returns[g] += np.mean(y_valid[sorted_idx[start:end]])
            group_counts[g] += 1

        top_ret = np.mean(y_valid[sorted_idx[-group_size:]])
        bot_ret = np.mean(y_valid[sorted_idx[:group_size]])
        spread_returns.append(top_ret - bot_ret)

    if np.min(group_counts) < 5 or len(spread_returns) < 5:
        return 0

    avg_returns = group_returns / (group_counts + 1e-10)
    mono_score = sum(
        0.25 if avg_returns[i + 1] > avg_returns[i]
        else 0.125 if avg_returns[i + 1] == avg_returns[i]
        else 0.0
        for i in range(n_groups - 1)
    )

    # 使用多空价差（top - bottom）替代绝对top收益
    # 这样在熊市中，只要因子能区分相对强弱，就能获得正fitness
    avg_spread = np.mean(spread_returns) * 252     # 年化多空价差
    spread_winrate = np.mean([1.0 if s > 0 else 0.0 for s in spread_returns])

    # 综合得分 = 多空价差 × 单调性权重 + 价差胜率加分
    score = avg_spread * (0.3 + mono_score * 0.7) + (spread_winrate - 0.5) * 2
    score = np.clip(score, -10, 10)
    return score if not np.isnan(score) else 0


# RSI 因子

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


def calc_rsi_combo_v2(close, rsi_period=9, mom_period=5):
    """RSI + 背离信号组合因子"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi_ema = 100 - (100 / (1 + rs))

    price_mom = close.pct_change(mom_period)
    rsi_mom = rsi_ema.diff(mom_period)

    bull_div = ((price_mom < -0.02) & (rsi_mom > 0) & (rsi_ema < 40)).astype(int)
    bear_div = ((price_mom > 0.02) & (rsi_mom < 0) & (rsi_ema > 60)).astype(int)
    return rsi_ema + bull_div * 80 - bear_div * 80


# 特征工程

def add_technical_features(x_dict, has_volume=True):
    close = x_dict['close']
    volume = x_dict['volume']
    ret = close.pct_change()

    new_features = {
        'RSI_COMBO_V2': calc_rsi_combo_v2(close, rsi_period=9, mom_period=5),
        'ACCEL_5': ret.diff(5),
        'PATH_MOM_20': ret.rolling(20).sum() / (ret.rolling(20).std() + 1e-10),
        'RET_20': close.pct_change(20),
        'VOL_20': ret.rolling(20).std(),
    }
    if has_volume:
        vol_change = volume / volume.rolling(10).mean()
        new_features['VP_ACCEL_10'] = (ret * vol_change).diff(10)

    for k, v in new_features.items():
        x_dict[k] = v.ffill().fillna(50 if 'RSI' in k else 0)
    return x_dict


# 因子后处理

def process_factor(factor_series, method='mad', n_sigma=3):
    """去极值 + Z-score 标准化"""
    factor = np.array(factor_series, dtype=np.float64)
    valid_mask = ~np.isnan(factor) & ~np.isinf(factor)
    if np.sum(valid_mask) < 10:
        return factor

    valid = factor[valid_mask]
    if method == 'mad':
        median = np.median(valid)
        mad_e = 1.4826 * np.median(np.abs(valid - median))
        lower, upper = median - n_sigma * mad_e, median + n_sigma * mad_e
    elif method == 'std':
        mu, sigma = np.mean(valid), np.std(valid)
        lower, upper = mu - n_sigma * sigma, mu + n_sigma * sigma
    elif method == 'percentile':
        lower, upper = np.percentile(valid, 1), np.percentile(valid, 99)
    else:
        raise ValueError(f"未知方法: {method}")

    clipped = np.clip(factor, lower, upper)
    valid_clipped = clipped[valid_mask]
    mu, sigma = np.mean(valid_clipped), np.std(valid_clipped)
    return (clipped - mu) / sigma if sigma > 1e-10 else clipped - mu


def process_factor_cross_section(factor_df, method='mad', n_sigma=3):
    result = factor_df.copy()
    for idx in result.index:
        result.loc[idx] = process_factor(result.loc[idx].values, method, n_sigma)
    return result


# 滚动训练核心

def rolling_factor_validation(close_df, x_dict,
                              train_window=120, pred_horizon=10,
                              rebalance_freq='M', gen_num=5, pop_num=500,
                              start_date=None, end_date=None, n_jobs=1):
    """滚动窗口GP因子挖掘，在有效期内逐日生成因子值"""
    print("=" * 60)
    print("滚动窗口因子有效性验证")
    print(f"训练窗口: {train_window}天 | 预测周期: {pred_horizon}天 | 调仓: {rebalance_freq}")
    print(f"GP参数: gen={gen_num}, pop={pop_num}")
    print("=" * 60)

    x_dict_processed = {k: v.rank(axis=1, pct=True).fillna(0.5)
                        for k, v in x_dict.items() if v is not None}
    feature_names = list(x_dict_processed.keys())
    all_days = close_df.index.tolist()

    # 生成调仓日
    if rebalance_freq == 'M':
        groups = close_df.groupby(pd.Grouper(freq='M'))
        eval_dates = pd.DatetimeIndex([g.index[-1] for _, g in groups if len(g) > 0])
    elif rebalance_freq == '2W':
        eval_dates = []
        for ym in pd.date_range(start=close_df.index[0], end=close_df.index[-1], freq='M'):
            dates = close_df.loc[str(ym)[:7]].index
            if len(dates) == 0:
                continue
            eval_dates.append(dates[-1])
            mid = dates[dates.day <= 15]
            if len(mid) > 0:
                eval_dates.append(mid[-1])
        eval_dates = pd.DatetimeIndex(sorted(set(eval_dates)))
    else:
        groups = close_df.groupby(pd.Grouper(freq='W'))
        eval_dates = pd.DatetimeIndex([g.index[-1] for _, g in groups if len(g) > 0])

    if start_date:
        eval_dates = eval_dates[eval_dates >= pd.Timestamp(start_date)]
    if end_date:
        eval_dates = eval_dates[eval_dates <= pd.Timestamp(end_date)]

    train_dates = [d for d in eval_dates if close_df.index.get_loc(d) >= train_window]
    print(f"训练日期数: {len(train_dates)}")

    factor_record_list = []
    factor_log = []

    for i, train_date in enumerate(tqdm(train_dates, desc="因子训练")):
        train_idx = close_df.index.get_loc(train_date)

        if i + 1 < len(train_dates):
            next_train_date = train_dates[i + 1]
        else:
            if end_date:
                max_end = pd.Timestamp(end_date) + pd.Timedelta(days=pred_horizon + 5)
                next_train_date = min(max_end, close_df.index[-1])
            else:
                next_train_date = close_df.index[-1]

        train_start = all_days[train_idx - train_window]
        train_end = all_days[train_idx - 1]

        price_train = close_df.loc[train_start:train_end]
        ret_train = price_train.shift(-pred_horizon) / price_train - 1
        y_train = ret_train.dropna(how='all')
        if len(y_train) < 20:
            continue

        x_dict_train = {}
        for key in feature_names:
            x_dict_train[key] = x_dict_processed[key].loc[train_start:train_end].reindex(y_train.index).fillna(0.5)

        x_array_train = np.array([x_dict_train[k].values for k in feature_names])
        x_array_train = np.transpose(x_array_train, axes=(1, 2, 0))
        x_array_train = np.nan_to_num(x_array_train, nan=0.0, posinf=1e9, neginf=-1e9)

        gp_model = None
        program_str = ''
        for retry in range(6):
            try:
                gp_model = my_gplearn(
                    FUNCTION_SET, score_func_ic,
                    feature_names=feature_names,
                    pop_num=pop_num, gen_num=gen_num,
                    parsimony_coefficient='auto',
                    const_range=(0.001, 1.0),
                    init_depth=(2, 3),
                    random_state=42 + (i % 10) * 100 + retry * 1000,
                    n_jobs=n_jobs
                )
                gp_model.fit(x_array_train, y_train.values)
                program = gp_model._program
                program_str = str(program)

                bad_formula = (
                    program.has_unary_nesting()
                    or program.length_ > 45
                    or program.depth_ > 6
                    or re.search(r'log\(\s*-\d', program_str) is not None
                )
                if bad_formula:
                    if retry < 5:
                        continue
                    gp_model = None
                    break
                break
            except Exception as e:
                print(f"  GP训练失败: {e}")
                gp_model = None
                break

        if gp_model is None:
            continue

        train_pred = gp_model.predict(x_array_train)
        pred_flat = train_pred.flatten()
        if np.nanstd(pred_flat) < 1e-10:
            continue

        train_ic = pd.DataFrame({'pred': pred_flat, 'true': y_train.values.flatten()}).corr().iloc[0, 1]
        if np.isnan(train_ic):
            continue

        factor_direction = 1 if train_ic >= 0 else -1
        print(f"  {train_date:%Y-%m-%d} | IC={train_ic:.4f} | dir={'+'if factor_direction>0 else'-'} | {program_str}")

        factor_log.append({
            '日期': train_date.strftime('%Y-%m-%d'),
            '因子公式': program_str,
            '训练IC': round(train_ic, 4),
            '多空方向': factor_direction
        })

        valid_period = close_df.loc[train_date:next_train_date].index
        if end_date:
            valid_period = valid_period[valid_period <= pd.Timestamp(end_date)]

        for eval_date in valid_period:
            eval_idx = close_df.index.get_loc(eval_date)
            current_feats = [x_dict_processed[k].loc[eval_date].values for k in feature_names]
            current_x = np.nan_to_num(np.array(current_feats).T, nan=0.5)
            pred_factor = process_factor(gp_model.predict(current_x), method='mad', n_sigma=3)
            pred_factor = pred_factor * factor_direction

            next_day_idx = eval_idx + 1
            if next_day_idx >= len(all_days):
                continue
            next_day = all_days[next_day_idx]
            next_ret = close_df.loc[next_day] / close_df.loc[eval_date] - 1

            temp_df = pd.DataFrame({
                'Time': eval_date, 'Tick': close_df.columns,
                'factor': pred_factor, 'next_return': next_ret.values
            }).dropna(subset=['factor', 'next_return'])
            factor_record_list.append(temp_df)

    if factor_record_list:
        factor_df = pd.concat(factor_record_list, ignore_index=True)
        factor_df = factor_df.drop_duplicates(subset=['Time', 'Tick'], keep='first')
    else:
        factor_df = pd.DataFrame()

    return factor_df, pd.DataFrame(factor_log)


# 数据加载

def load_industry_index_data(industry_level='level1'):
    """加载中信行业指数OHLCV宽表数据"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, '..', '..', '中信一级行业')

    level_dirs = {
        'level1': os.path.join(base_path, '中信一级'),
        'level2': os.path.join(base_path, '中信二级', '中信二级'),
        'level3': os.path.join(base_path, '中信三级'),
    }
    level_names = {'level1': '中信一级', 'level2': '中信二级', 'level3': '中信三级'}

    data_dir = level_dirs.get(industry_level)
    if data_dir is None or not os.path.exists(data_dir):
        raise ValueError(f"目录不存在: {data_dir}")

    csv_files = glob.glob(os.path.join(data_dir, '*.CSV'))
    if not csv_files:
        raise FileNotFoundError(f"未找到CSV: {data_dir}")

    all_data = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding='gbk')
            df.columns = df.columns.str.strip()
            all_data.append(df)
        except Exception as e:
            print(f"  读取失败: {f} ({e})")

    combined = pd.concat(all_data, ignore_index=True)
    col_map = {'日期': 'date', '代码': 'code', '开盘价': 'open', '最高价': 'high',
               '最低价': 'low', '收盘价': 'close', '成交量(股)': 'volume'}
    combined = combined.rename(columns=col_map)
    combined['date'] = pd.to_datetime(combined['date'])
    keep = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
    combined = combined[[c for c in keep if c in combined.columns]]

    kw = dict(index='date', columns='code')
    close_data = combined.pivot(values='close', **kw).sort_index()
    open_data = combined.pivot(values='open', **kw).sort_index()
    high_data = combined.pivot(values='high', **kw).sort_index()
    low_data = combined.pivot(values='low', **kw).sort_index()
    volume_data = combined.pivot(values='volume', **kw).sort_index()

    x_dict = {
        'open': open_data, 'high': high_data, 'low': low_data,
        'close': close_data, 'volume': volume_data,
        'total_turnover': volume_data * close_data
    }

    print(f"{level_names[industry_level]}: {len(close_data.columns)}个行业, "
          f"{close_data.index[0]:%Y-%m-%d} ~ {close_data.index[-1]:%Y-%m-%d}")
    return close_data, x_dict, True


# 主程序

if __name__ == '__main__':
    INDUSTRY_LEVEL = 'level1'

    try:
        close_data, x_dict, has_volume = load_industry_index_data(INDUSTRY_LEVEL)
        GP_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'total_turnover']
        x_dict_for_gp = {k: v for k, v in x_dict.items() if k in GP_FEATURES}
    except Exception as e:
        print(f"数据加载失败: {e}")
        exit()

    # 检查数据覆盖范围
    data_end = close_data.index[-1]
    print(f"\n数据实际覆盖: {close_data.index[0]:%Y-%m-%d} ~ {data_end:%Y-%m-%d}")

    CONFIG = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'train_window': 300,
        'pred_horizon': 30,
        'rebalance_freq': 'M',
        'gen_num': 4,
        'pop_num': 2000,
        'n_jobs': -1
    }

    cfg_end = pd.Timestamp(CONFIG['end_date'])
    if cfg_end > data_end:
        print(f"\n⚠ 警告: end_date({CONFIG['end_date']})超出数据范围({data_end:%Y-%m-%d})!")
        print(f"  训练将在数据结束日自动截止。请更新行业指数CSV数据至{CONFIG['end_date']}。")

    factor_df, factor_log_df = rolling_factor_validation(
        close_df=close_data, x_dict=x_dict_for_gp, **CONFIG
    )

    if not factor_df.empty:
        factor_df.to_csv(f'{RESULT_DIR}factor_data.csv', index=False, encoding='utf-8-sig')
        print(f"\n因子数据: {len(factor_df)}条, {factor_df['Time'].nunique()}个交易日")

    if not factor_log_df.empty:
        factor_log_df.to_excel(f'{RESULT_DIR}factor_log.xlsx', index=False)

    if not factor_df.empty:
        fa = FactorAnalysis(factor_df, factor_col='factor',
                            return_col='next_return', time_col='Time')
        ic_df, group_ret, cum_ret, monthly_ic, ls_ret = fa.run_full_analysis(n_groups=5)

        ic_df.to_csv(f'{RESULT_DIR}ic_series.csv')
        group_ret.to_csv(f'{RESULT_DIR}group_returns.csv')
        cum_ret.to_csv(f'{RESULT_DIR}cumulative_returns.csv')

        try:
            fa.plot_analysis(ic_df, group_ret, cum_ret, monthly_ic, ls_ret)
            plt.savefig(f'{RESULT_DIR}factor_analysis_report.png', dpi=150, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"绘图失败: {e}")

        print(f"\n训练期数: {len(factor_log_df)}, 平均IC: {factor_log_df['训练IC'].mean():.4f}")
        for _, row in factor_log_df.iterrows():
            print(f"  {row['日期']}: {row['因子公式']} (IC={row['训练IC']})")
