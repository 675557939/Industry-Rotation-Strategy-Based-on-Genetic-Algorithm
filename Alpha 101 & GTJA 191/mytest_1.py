import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def winsorize_standardize(series, lower_q=0.025, upper_q=0.975):
    lo, hi = series.quantile(lower_q), series.quantile(upper_q)
    series = series.clip(lower=lo, upper=hi)
    std = series.std()
    return (series - series.mean()) / std if std != 0 else series - series.mean()


def calculate_ic_metrics(alpha_df, next_return_series):
    merged = pd.merge(alpha_df, next_return_series, on=['date', 'idx'], how='inner').dropna()
    if merged.empty:
        return None, None

    ic_data = []
    for date, group in merged.groupby('date'):
        if len(group) > 1:
            ic, p_val = spearmanr(group['alpha_value'], group['next_return'])
            ic_data.append({'date': date, 'IC': ic, 'p_value': p_val})
        else:
            ic_data.append({'date': date, 'IC': np.nan, 'p_value': np.nan})

    ic_df = pd.DataFrame(ic_data).set_index('date').dropna()
    if ic_df.empty:
        return {k: np.nan for k in ['平均IC', 'IC标准差', 'ICIR', 'IC胜率', '显著IC胜率(p<0.05)', '自相关性']}, ic_df

    avg_ic = ic_df['IC'].mean()
    std_ic = ic_df['IC'].std()

    return {
        '平均IC': avg_ic,
        'IC标准差': std_ic,
        'ICIR': avg_ic / std_ic if std_ic != 0 else np.nan,
        'IC胜率': (ic_df['IC'] > 0).mean(),
        '显著IC胜率(p<0.05)': (ic_df['p_value'] < 0.05).mean(),
        '自相关性': ic_df['IC'].autocorr(lag=1)
    }, ic_df


def main():
    alpha_data = pd.read_csv('alpha101.csv')
    alpha_data['date'] = pd.to_datetime(alpha_data['date'])

    original = pd.read_csv('CI指数_日线数据.csv')
    original['date'] = pd.to_datetime(original['date'])
    original.rename(columns={'order_book_id': 'idx'}, inplace=True)
    original['next_return'] = original.groupby('idx')['close'].pct_change().shift(-1)
    next_ret = original[['date', 'idx', 'next_return']]

    alpha_cols = [c for c in alpha_data.columns if c.startswith('alpha')]
    skip = {'alpha061', 'alpha075', 'alpha095'}
    all_metrics = {}

    for col in alpha_cols:
        if col in skip:
            continue
        temp = alpha_data[['date', 'idx', col]].copy().rename(columns={col: 'alpha_value'})
        temp['alpha_value'] = temp.groupby('date')['alpha_value'].transform(winsorize_standardize)
        metrics, _ = calculate_ic_metrics(temp, next_ret)
        if metrics:
            all_metrics[col] = metrics

    if all_metrics:
        results = pd.DataFrame(all_metrics)
        results.to_csv('alpha_performance_metrics.csv', encoding='utf-8-sig')
        print(results)


if __name__ == '__main__':
    main()
