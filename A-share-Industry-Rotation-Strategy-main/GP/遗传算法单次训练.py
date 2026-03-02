# -*- coding: utf-8 -*-
"""
行业轮动因子挖掘 - 单次训练（80%训练 + 20%验证）
在历史数据前80%训练GP因子，后20%做样本外验证。
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
FITNESS_MODE = 'rank_icir'


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
    mean_x, mean_y = sum_x / n, sum_y / n
    cov, var_x, var_y = 0.0, 0.0, 0.0
    for i in range(n):
        dx, dy = x[i] - mean_x, y[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    if var_x < 1e-10 or var_y < 1e-10:
        return 0.0
    return cov / (np.sqrt(var_x) * np.sqrt(var_y))


@jit(nopython=True, parallel=True)
def _spearman_ic_2d(y_true, y_pred):
    n_days, n_stocks = y_true.shape
    ics = np.zeros(n_days, dtype=np.float64)
    for day in prange(n_days):
        y_row, pred_row = y_true[day, :], y_pred[day, :]
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
                y_valid[idx], pred_valid[idx] = y_row[j], pred_row[j]
                idx += 1
        ics[day] = _corr_1d(_rankdata_1d(y_valid), _rankdata_1d(pred_valid))
    return ics


# 冗余检测

def has_redundant_nesting(program_str):
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
    return sum(program_str.count(op + '(') for op in ['add', 'sub', 'mul', 'div', 'max', 'min'])


# 适应度函数

def score_func_ic(y, y_pred, sample_weight):
    if len(np.unique(y_pred[-1])) <= 10:
        return -1
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    pred_arr = np.ascontiguousarray(y_pred, dtype=np.float64)
    if FITNESS_MODE == 'rank_icir':
        return _fitness_rank_icir(y_arr, pred_arr)
    elif FITNESS_MODE == 'mono_topret':
        return _fitness_mono_topret(y_arr, pred_arr)
    return _fitness_ic_winrate(y_arr, pred_arr)


def _calc_group_returns(y_true, y_pred, n_groups=5):
    n_days = y_true.shape[0]
    gr = np.zeros(n_groups, dtype=np.float64)
    gc = np.zeros(n_groups, dtype=np.float64)
    for day in range(n_days):
        y_row, pred_row = y_true[day, :], y_pred[day, :]
        mask = ~(np.isnan(y_row) | np.isnan(pred_row))
        if np.sum(mask) < n_groups * 3:
            continue
        y_v, p_v = y_row[mask], pred_row[mask]
        si = np.argsort(p_v)
        gs = len(si) // n_groups
        for g in range(n_groups):
            s = g * gs
            e = s + gs if g < n_groups - 1 else len(si)
            gr[g] += np.mean(y_v[si[s:e]])
            gc[g] += 1
    if np.min(gc) < 1:
        return None
    return gr / (gc + 1e-10)


def _fitness_rank_icir(y_true, y_pred):
    ics = _spearman_ic_2d(y_true, y_pred)
    valid = ics[~np.isnan(ics)]
    if len(valid) < 5:
        return 0
    ic_mean = np.mean(valid)
    ic_std = np.std(valid, ddof=1)
    icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
    score = np.clip(icir * 10, -10, 10)
    return score if not np.isnan(score) else 0


def _fitness_ic_winrate(y_true, y_pred, n_groups=5):
    ics = _spearman_ic_2d(y_true, y_pred)
    valid = ics[~np.isnan(ics)]
    if len(valid) < 5:
        return 0
    ic_mean = np.mean(valid)
    win_rate = np.mean(np.array([1.0 if ic > 0 else 0.0 for ic in valid]))
    if ic_mean < 0:
        win_rate = 1 - win_rate
    base = win_rate * abs(ic_mean) * 100
    avg_ret = _calc_group_returns(y_true, y_pred, n_groups)
    if avg_ret is None:
        return base * 0.3
    checks = [avg_ret[i] < avg_ret[i+1] for i in range(n_groups-1)]
    w = [0.20, 0.15, 0.15, 0.20]
    p = [-0.40, -0.20, -0.15, -0.30]
    lb = sum(wi if c else pi for c, wi, pi in zip(checks, w, p))
    top_ann = avg_ret[-1] * 252
    spread_ann = (avg_ret[-1] - avg_ret[0]) * 252
    rb = (min(top_ann*0.5, 0.3) if top_ann > 0 else -0.3) + (min(spread_ann*0.3, 0.3) if spread_ann > 0 else -0.2)
    return np.clip(base * (1 + lb + rb), -10, 10)


def _fitness_mono_topret(y_true, y_pred, n_groups=5):
    n_days = y_true.shape[0]
    gr = np.zeros(n_groups, dtype=np.float64)
    gc = np.zeros(n_groups, dtype=np.float64)
    top_rets = []
    for day in range(n_days):
        y_row, pred_row = y_true[day, :], y_pred[day, :]
        mask = ~(np.isnan(y_row) | np.isnan(pred_row))
        if np.sum(mask) < n_groups * 3:
            continue
        y_v, p_v = y_row[mask], pred_row[mask]
        si = np.argsort(p_v)
        gs = len(si) // n_groups
        for g in range(n_groups):
            s = g * gs
            e = s + gs if g < n_groups - 1 else len(si)
            gr[g] += np.mean(y_v[si[s:e]])
            gc[g] += 1
        top_rets.append(np.mean(y_v[si[-gs:]]))
    if np.min(gc) < 5 or len(top_rets) < 5:
        return 0
    avg_ret = gr / (gc + 1e-10)
    mono = sum(0.25 if avg_ret[i+1] > avg_ret[i] else 0.125 if avg_ret[i+1] == avg_ret[i] else 0.0
               for i in range(n_groups - 1))
    return np.clip(np.mean(top_rets) * 252 * (0.3 + mono * 0.7), -10, 10)


# RSI 因子

def calc_rsi_combo_v2(close, rsi_period=9, mom_period=5):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
    rsi_ema = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-6))
    price_mom = close.pct_change(mom_period)
    rsi_mom = rsi_ema.diff(mom_period)
    bull = ((price_mom < -0.02) & (rsi_mom > 0) & (rsi_ema < 40)).astype(int)
    bear = ((price_mom > 0.02) & (rsi_mom < 0) & (rsi_ema > 60)).astype(int)
    return rsi_ema + bull * 80 - bear * 80


# 特征工程

def add_technical_features(x_dict, has_volume=True):
    close, volume = x_dict['close'], x_dict['volume']
    ret = close.pct_change()
    feats = {
        'RSI_COMBO_V2': calc_rsi_combo_v2(close),
        'ACCEL_5': ret.diff(5),
        'PATH_MOM_20': ret.rolling(20).sum() / (ret.rolling(20).std() + 1e-10),
        'RET_20': close.pct_change(20),
        'VOL_20': ret.rolling(20).std(),
    }
    if has_volume:
        feats['VP_ACCEL_10'] = (ret * volume / volume.rolling(10).mean()).diff(10)
    for k, v in feats.items():
        x_dict[k] = v.ffill().fillna(50 if 'RSI' in k else 0)
    return x_dict


# 因子后处理

def process_factor(factor_series, method='mad', n_sigma=3):
    factor = np.array(factor_series, dtype=np.float64)
    valid_mask = ~np.isnan(factor) & ~np.isinf(factor)
    if np.sum(valid_mask) < 10:
        return factor
    valid = factor[valid_mask]
    if method == 'mad':
        med = np.median(valid)
        mad_e = 1.4826 * np.median(np.abs(valid - med))
        lo, hi = med - n_sigma * mad_e, med + n_sigma * mad_e
    elif method == 'std':
        mu, sig = np.mean(valid), np.std(valid)
        lo, hi = mu - n_sigma * sig, mu + n_sigma * sig
    else:
        lo, hi = np.percentile(valid, 1), np.percentile(valid, 99)
    clipped = np.clip(factor, lo, hi)
    vc = clipped[valid_mask]
    mu, sig = np.mean(vc), np.std(vc)
    return (clipped - mu) / sig if sig > 1e-10 else clipped - mu


# 单次训练核心

def train_validate_factor(close_df, x_dict, train_ratio=0.8,
                          pred_horizon=10, gen_num=5, pop_num=2000,
                          start_date=None, end_date=None, n_jobs=1):
    """前 train_ratio 训练 GP，后面做样本外验证"""
    print('=' * 60)
    print(f'单次训练验证 | 训练比例={train_ratio} | 预测周期={pred_horizon}天')
    print(f'GP: gen={gen_num}, pop={pop_num}')
    print('=' * 60)

    if start_date:
        close_df = close_df.loc[start_date:]
    if end_date:
        close_df = close_df.loc[:end_date]

    x_dict_proc = {k: v.rank(axis=1, pct=True).fillna(0.5)
                   for k, v in x_dict.items() if v is not None}
    feature_names = list(x_dict_proc.keys())

    n_total = len(close_df)
    split_idx = int(n_total * train_ratio)
    train_dates = close_df.index[:split_idx]
    test_dates = close_df.index[split_idx:]
    print(f'训练: {train_dates[0]:%Y-%m-%d}~{train_dates[-1]:%Y-%m-%d} ({len(train_dates)}天)')
    print(f'测试: {test_dates[0]:%Y-%m-%d}~{test_dates[-1]:%Y-%m-%d} ({len(test_dates)}天)')

    # 训练集
    price_train = close_df.loc[train_dates[0]:train_dates[-1]]
    ret_train = price_train.shift(-pred_horizon) / price_train - 1
    y_train = ret_train.dropna(how='all')

    x_train = {}
    for k in feature_names:
        x_train[k] = x_dict_proc[k].reindex(y_train.index).fillna(0.5)

    x_arr = np.nan_to_num(
        np.transpose(np.array([x_train[k].values for k in feature_names]), (1, 2, 0)),
        nan=0.0, posinf=1e9, neginf=-1e9
    )

    # GP训练
    gp_model = None
    for retry in range(3):
        try:
            gp_model = my_gplearn(
                FUNCTION_SET, score_func_ic,
                feature_names=feature_names,
                pop_num=pop_num, gen_num=gen_num,
                random_state=42 + retry * 1000, n_jobs=n_jobs
            )
            gp_model.fit(x_arr, y_train.values)
            program_str = str(gp_model._program)
            if has_redundant_nesting(program_str) and count_binary_ops(program_str) < 2 and retry < 2:
                continue
            break
        except Exception as e:
            print(f'训练失败: {e}')
            gp_model = None

    if gp_model is None:
        print('GP训练彻底失败')
        return pd.DataFrame(), pd.DataFrame()

    program_str = str(gp_model._program)
    train_pred = gp_model.predict(x_arr)
    train_ic = pd.DataFrame({'p': train_pred.flatten(), 't': y_train.values.flatten()}).corr().iloc[0, 1]
    factor_direction = 1 if train_ic >= 0 else -1
    print(f'\n因子: {program_str}')
    print(f'训练IC: {train_ic:.4f} | 方向: {"+" if factor_direction > 0 else "-"}')

    # 全期逐日生成因子值
    all_days = close_df.index.tolist()
    records = []
    for eval_date in tqdm(close_df.index, desc='逐日计算因子'):
        eval_idx = close_df.index.get_loc(eval_date)
        feats = [x_dict_proc[k].loc[eval_date].values for k in feature_names]
        current_x = np.nan_to_num(np.array(feats).T, nan=0.5)
        pred = process_factor(gp_model.predict(current_x)) * factor_direction

        ndi = eval_idx + 1
        if ndi >= len(all_days):
            continue
        next_ret = close_df.loc[all_days[ndi]] / close_df.loc[eval_date] - 1

        records.append(pd.DataFrame({
            'Time': eval_date, 'Tick': close_df.columns,
            'factor': pred, 'next_return': next_ret.values
        }).dropna(subset=['factor', 'next_return']))

    factor_df = pd.concat(records, ignore_index=True).drop_duplicates(subset=['Time', 'Tick'], keep='first') if records else pd.DataFrame()
    log_df = pd.DataFrame([{'因子公式': program_str, '训练IC': round(train_ic, 4), '多空方向': factor_direction}])
    return factor_df, log_df


# 数据加载

def load_industry_index_data(industry_level='level1'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, '..', '..', '中信一级行业')
    level_dirs = {
        'level1': os.path.join(base_path, '中信一级'),
        'level2': os.path.join(base_path, '中信二级', '中信二级'),
        'level3': os.path.join(base_path, '中信三级'),
    }
    names = {'level1': '中信一级', 'level2': '中信二级', 'level3': '中信三级'}
    data_dir = level_dirs.get(industry_level)
    if not data_dir or not os.path.exists(data_dir):
        raise ValueError(f"目录不存在: {data_dir}")

    csv_files = glob.glob(os.path.join(data_dir, '*.CSV'))
    if not csv_files:
        raise FileNotFoundError(f"未找到CSV: {data_dir}")

    frames = []
    for f in csv_files:
        try:
            d = pd.read_csv(f, encoding='gbk')
            d.columns = d.columns.str.strip()
            frames.append(d)
        except Exception as e:
            print(f"  读取失败: {f} ({e})")

    combined = pd.concat(frames, ignore_index=True)
    col_map = {'日期': 'date', '代码': 'code', '开盘价': 'open', '最高价': 'high',
               '最低价': 'low', '收盘价': 'close', '成交量(股)': 'volume'}
    combined = combined.rename(columns=col_map)
    combined['date'] = pd.to_datetime(combined['date'])
    keep = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
    combined = combined[[c for c in keep if c in combined.columns]]

    kw = dict(index='date', columns='code')
    close_data = combined.pivot(values='close', **kw).sort_index()
    x_dict = {
        'open': combined.pivot(values='open', **kw).sort_index(),
        'high': combined.pivot(values='high', **kw).sort_index(),
        'low': combined.pivot(values='low', **kw).sort_index(),
        'close': close_data,
        'volume': combined.pivot(values='volume', **kw).sort_index(),
        'total_turnover': combined.pivot(values='volume', **kw).sort_index() * close_data
    }
    print(f"{names[industry_level]}: {len(close_data.columns)}个行业, "
          f"{close_data.index[0]:%Y-%m-%d} ~ {close_data.index[-1]:%Y-%m-%d}")
    return close_data, x_dict, True


# 主程序

if __name__ == '__main__':
    try:
        close_data, x_dict, _ = load_industry_index_data('level1')
        GP_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'total_turnover']
        x_dict_gp = {k: v for k, v in x_dict.items() if k in GP_FEATURES}
    except Exception as e:
        print(f"数据加载失败: {e}")
        exit()

    factor_df, log_df = train_validate_factor(
        close_data, x_dict_gp,
        train_ratio=0.8, pred_horizon=30,
        gen_num=5, pop_num=2000,
        start_date='2015-01-01', end_date='2025-12-31',
        n_jobs=3
    )

    if not factor_df.empty:
        factor_df.to_csv(f'{RESULT_DIR}factor_data_single.csv', index=False, encoding='utf-8-sig')
        fa = FactorAnalysis(factor_df, factor_col='factor', return_col='next_return', time_col='Time')
        results = fa.run_full_analysis(n_groups=5)
        try:
            fa.plot_analysis(*results)
            plt.savefig(f'{RESULT_DIR}single_train_report.png', dpi=150, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"绘图失败: {e}")
