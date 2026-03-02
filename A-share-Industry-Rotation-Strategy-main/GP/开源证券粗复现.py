# -*- coding: utf-8 -*-
"""
行业轮动因子挖掘 - 开源证券GP框架复现
参考：开源证券《市场微观结构研究系列（20）：遗传算法赋能交易行为因子》
80%训练 + 20%验证，使用Rank ICIR适应度，含因子过滤。
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

# GP函数集（基础运算 + 时序算子 + 截面算子）
FUNCTION_SET = [
    'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs',
    'max', 'min', 'neg', 'inv',
    'delayy', 'delta',
    'ts_max_5', 'ts_max_20', 'ts_min_5', 'ts_min_20',
    'ts_std_5', 'ts_std_20', 'decayl',
    'rankk', 'signedpower',
]

# 因子过滤阈值
IC_THRESHOLD = 0.02
ICIR_THRESHOLD = 0.3
CORR_THRESHOLD = 0.7
MIN_COMPLEXITY = 3
MAX_COMPLEXITY = 20


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


# 冗余与复杂度检测

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


def count_operators(program_str):
    ops = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'max', 'min',
           'neg', 'inv', 'delayy', 'delta', 'ts_max', 'ts_min', 'ts_std',
           'decayl', 'rankk', 'signedpower']
    return sum(program_str.count(op + '(') for op in ops)


def calc_factor_correlation(factor1, factor2):
    f1, f2 = factor1.values.flatten(), factor2.values.flatten()
    mask = ~(np.isnan(f1) | np.isnan(f2))
    if np.sum(mask) < 100:
        return np.nan
    return np.corrcoef(f1[mask], f2[mask])[0, 1]


def filter_factor(factor_values, ic_mean, icir, program_str, existing_factors=None):
    """因子过滤：IC/ICIR阈值、复杂度、冗余、相关性"""
    if abs(ic_mean) < IC_THRESHOLD:
        return False, f'|IC|={abs(ic_mean):.4f} < {IC_THRESHOLD}'
    if abs(icir) < ICIR_THRESHOLD:
        return False, f'|ICIR|={abs(icir):.4f} < {ICIR_THRESHOLD}'
    complexity = count_operators(program_str)
    if complexity < MIN_COMPLEXITY:
        return False, f'复杂度{complexity} < {MIN_COMPLEXITY}'
    if complexity > MAX_COMPLEXITY:
        return False, f'复杂度{complexity} > {MAX_COMPLEXITY}'
    if has_redundant_nesting(program_str):
        return False, '存在冗余嵌套'
    if existing_factors is not None:
        for name, ef in existing_factors.items():
            corr = calc_factor_correlation(factor_values, ef)
            if not np.isnan(corr) and abs(corr) > CORR_THRESHOLD:
                return False, f'与{name}相关系数{corr:.4f} > {CORR_THRESHOLD}'
    return True, '通过'


# 适应度函数

def score_func_ic(y, y_pred, sample_weight):
    if len(np.unique(y_pred[-1])) <= 5:
        return -1
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    pred_arr = np.ascontiguousarray(y_pred, dtype=np.float64)
    ics = _spearman_ic_2d(y_arr, pred_arr)
    valid = ics[~np.isnan(ics)]
    if len(valid) < 10:
        return 0
    ic_mean, ic_std = np.mean(valid), np.std(valid)
    if ic_std < 1e-10:
        return 0
    score = np.clip(abs(ic_mean / ic_std) * 10, 0, 15)
    return score if not np.isnan(score) else 0


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


# 80%训练 + 20%验证

def train_validate_factor(close_df, x_dict, train_ratio=0.8,
                          pred_horizon=10, gen_num=5, pop_num=500,
                          start_date=None, end_date=None, n_jobs=1):
    """前train_ratio训练GP，后面做样本外验证，返回(train_df, valid_df, log_df, stats)"""
    print('=' * 60)
    print(f'开源证券GP训练 | 训练{train_ratio*100:.0f}%/验证{(1-train_ratio)*100:.0f}%')
    print(f'预测周期={pred_horizon}天 | gen={gen_num}, pop={pop_num}')
    print('=' * 60)

    # 特征预处理：截面排名标准化
    x_dict_proc = {k: v.rank(axis=1, pct=True).fillna(0.5)
                   for k, v in x_dict.items() if v is not None}
    feature_names = list(x_dict_proc.keys())

    if start_date:
        close_df = close_df[close_df.index >= pd.Timestamp(start_date)]
        for k in x_dict_proc:
            x_dict_proc[k] = x_dict_proc[k][x_dict_proc[k].index >= pd.Timestamp(start_date)]
    if end_date:
        close_df = close_df[close_df.index <= pd.Timestamp(end_date)]
        for k in x_dict_proc:
            x_dict_proc[k] = x_dict_proc[k][x_dict_proc[k].index <= pd.Timestamp(end_date)]

    all_days = close_df.index.tolist()
    n_days = len(all_days)
    split_idx = int(n_days * train_ratio)
    train_end_idx = split_idx - pred_horizon

    train_dates = all_days[:train_end_idx]
    valid_dates = all_days[split_idx:]
    print(f'训练: {train_dates[0]:%Y-%m-%d}~{train_dates[-1]:%Y-%m-%d} ({len(train_dates)}天)')
    print(f'验证: {valid_dates[0]:%Y-%m-%d}~{valid_dates[-1]:%Y-%m-%d} ({len(valid_dates)}天)')

    if len(train_dates) < 60:
        raise ValueError(f'训练集不足: {len(train_dates)}天 < 60天')
    if len(valid_dates) < 20:
        raise ValueError(f'验证集不足: {len(valid_dates)}天 < 20天')

    # 训练数据
    price_train = close_df.loc[train_dates[0]:train_dates[-1]]
    ret_train = price_train.shift(-pred_horizon) / price_train - 1
    y_train = ret_train.dropna(how='all')
    x_train = {k: x_dict_proc[k].reindex(y_train.index).fillna(0.5) for k in feature_names}
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
            complexity = count_operators(program_str)
            if has_redundant_nesting(program_str) and complexity < MIN_COMPLEXITY and retry < 2:
                continue
            break
        except Exception as e:
            print(f'训练失败: {e}')
            gp_model = None

    if gp_model is None:
        raise RuntimeError('GP训练失败')

    program_str = str(gp_model._program)
    train_pred = gp_model.predict(x_arr)
    train_ic = pd.DataFrame({'p': train_pred.flatten(), 't': y_train.values.flatten()}).corr().iloc[0, 1]
    factor_direction = 1 if train_ic >= 0 else -1
    print(f'\n因子: {program_str}')
    print(f'训练IC: {train_ic:.4f} | 方向: {"+" if factor_direction > 0 else "-"}')

    factor_log = [{'训练区间': f'{train_dates[0]:%Y-%m-%d}~{train_dates[-1]:%Y-%m-%d}',
                   '验证区间': f'{valid_dates[0]:%Y-%m-%d}~{valid_dates[-1]:%Y-%m-%d}',
                   '因子公式': program_str, '训练IC': round(train_ic, 4),
                   '多空方向': factor_direction}]

    # 逐日生成因子值（公用逻辑）
    def _gen_factor(dates, label):
        records = []
        for eval_date in tqdm(dates, desc=f'{label}因子计算'):
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
                'factor': pred, 'next_return': next_ret.values, 'dataset': label
            }).dropna(subset=['factor', 'next_return']))
        return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

    train_df = _gen_factor(train_dates, 'train')
    valid_df = _gen_factor(valid_dates, 'valid')

    # IC统计
    analysis = {}
    for name, df in [('train', train_df), ('valid', valid_df)]:
        if df.empty:
            continue
        daily_ic = df.groupby('Time').apply(lambda x: x['factor'].corr(x['next_return'], method='spearman'))
        ic_mean, ic_std = daily_ic.mean(), daily_ic.std()
        icir = ic_mean / (ic_std + 1e-10)
        analysis[name] = {'IC均值': ic_mean, 'ICIR': icir, 'IC胜率': (daily_ic > 0).mean()}
        print(f'[{name}] IC={ic_mean:.4f}, ICIR={icir:.4f}')

    # 收敛性检验
    if 'train' in analysis and 'valid' in analysis:
        same_dir = (analysis['valid']['IC均值'] * analysis['train']['IC均值']) > 0
        valid_ok = same_dir and abs(analysis['valid']['ICIR']) > 0.05
        analysis['convergence'] = {'IC同向': same_dir, '因子有效': valid_ok}
        print(f'收敛性: IC同向={same_dir}, 有效={valid_ok}')

    # 开源证券因子过滤
    passed, reason = filter_factor(
        train_df, train_ic,
        analysis.get('train', {}).get('ICIR', 0),
        program_str, existing_factors=None
    )
    analysis['filter'] = {'passed': passed, 'reason': reason, 'complexity': count_operators(program_str)}
    print(f'过滤: {"通过" if passed else reason}')

    return train_df, valid_df, pd.DataFrame(factor_log), analysis


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
        raise ValueError(f'目录不存在: {data_dir}')

    csv_files = glob.glob(os.path.join(data_dir, '*.CSV'))
    if not csv_files:
        raise FileNotFoundError(f'未找到CSV: {data_dir}')

    frames = []
    for f in csv_files:
        try:
            d = pd.read_csv(f, encoding='gbk')
            d.columns = d.columns.str.strip()
            frames.append(d)
        except Exception as e:
            print(f'读取失败: {f} ({e})')

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
    print(f'{names[industry_level]}: {len(close_data.columns)}个行业, '
          f'{close_data.index[0]:%Y-%m-%d} ~ {close_data.index[-1]:%Y-%m-%d}')
    return close_data, x_dict, True


# 主程序

if __name__ == '__main__':
    try:
        close_data, x_dict, has_volume = load_industry_index_data('level1')
        x_dict = add_technical_features(x_dict, has_volume=has_volume)
        GP_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'total_turnover']
        x_dict_gp = {k: v for k, v in x_dict.items() if k in GP_FEATURES}
    except Exception as e:
        print(f'数据加载失败: {e}')
        exit()

    CONFIG = {
        'start_date': '2015-01-01',
        'end_date': '2025-12-31',
        'train_ratio': 0.8,
        'pred_horizon': 20,
        'gen_num': 10,
        'pop_num': 1000,
        'n_jobs': -1,
    }

    train_df, valid_df, log_df, results = train_validate_factor(
        close_data, x_dict_gp, **CONFIG
    )

    # 保存结果
    all_df = pd.concat([train_df, valid_df], ignore_index=True)
    if not all_df.empty:
        all_df.to_csv(f'{RESULT_DIR}factor_data.csv', index=False, encoding='utf-8-sig')
    if not train_df.empty:
        train_df.to_csv(f'{RESULT_DIR}factor_data_train.csv', index=False, encoding='utf-8-sig')
    if not valid_df.empty:
        valid_df.to_csv(f'{RESULT_DIR}factor_data_valid.csv', index=False, encoding='utf-8-sig')
    if not log_df.empty:
        log_df.to_excel(f'{RESULT_DIR}factor_log.xlsx', index=False)

    # 分组分析
    for label, df in [('训练集', train_df), ('验证集', valid_df)]:
        if df.empty:
            continue
        print(f'\n{"="*20} {label}分析 {"="*20}')
        fa = FactorAnalysis(df, factor_col='factor', return_col='next_return', time_col='Time')
        ic_df, group_ret, cum_ret, monthly_ic, ls_ret = fa.run_full_analysis(n_groups=5)
        suffix = 'train' if label == '训练集' else 'valid'
        ic_df.to_csv(f'{RESULT_DIR}ic_series_{suffix}.csv')
        group_ret.to_csv(f'{RESULT_DIR}group_returns_{suffix}.csv')
        cum_ret.to_csv(f'{RESULT_DIR}cumulative_returns_{suffix}.csv')
        try:
            fa.plot_analysis(ic_df, group_ret, cum_ret, monthly_ic, ls_ret)
            plt.suptitle(f'{label}因子分析', fontsize=14, y=1.02)
            plt.savefig(f'{RESULT_DIR}factor_analysis_{suffix}.png', dpi=150, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f'{label}绘图失败: {e}')
