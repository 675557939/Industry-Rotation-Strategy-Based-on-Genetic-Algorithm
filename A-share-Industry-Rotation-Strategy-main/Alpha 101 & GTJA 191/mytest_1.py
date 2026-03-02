import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def winsorize_standardize(series, lower_quantile=0.025, upper_quantile=0.975):
    """
    對數據進行去極值和標準化處理
    """
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)
    series = series.clip(lower=lower_bound, upper=upper_bound)
    std = series.std()
    if std == 0:
        return series - series.mean()
    return (series - series.mean()) / std


def calculate_ic_metrics(alpha_df, next_return_series):
    """
    計算IC系列及其各項指標

    參數：
    alpha_df (pd.DataFrame): 包含一個alpha因子的DataFrame
    next_return_series (pd.Series): 次日回報率序列

    返回：
    dict: 包含各種IC指標的字典
    """
    # 合併因子和回報數據
    merged_df = pd.merge(alpha_df, next_return_series, on=['date', 'idx'], how='inner')

    # 刪除缺失值
    merged_df.dropna(inplace=True)

    if merged_df.empty:
        return None, None

    # 按日期分組計算每日IC值和p值
    ic_data = []
    for date, group in merged_df.groupby('date'):
        if len(group) > 1:
            try:
                # 計算Spearman相關係數
                ic, p_value = spearmanr(group['alpha_value'], group['next_return'])
                ic_data.append({'date': date, 'IC': ic, 'p_value': p_value})
            except Exception as e:
                # 處理所有可能的錯誤，例如數據全部相同
                print(f"在 {date} 日期計算IC時出錯: {e}")
                ic_data.append({'date': date, 'IC': np.nan, 'p_value': np.nan})
        else:
            ic_data.append({'date': date, 'IC': np.nan, 'p_value': np.nan})

    ic_df = pd.DataFrame(ic_data).set_index('date')
    ic_df.dropna(inplace=True)

    if ic_df.empty:
        return {
            '平均IC': np.nan,
            'IC标准差': np.nan,
            'ICIR': np.nan,
            'IC胜率': np.nan,
            '显著IC胜率(p<0.05)': np.nan,
            '自相关性': np.nan
        }, ic_df

    # 計算各項指標
    avg_ic = ic_df['IC'].mean()
    std_ic = ic_df['IC'].std()
    icir = avg_ic / std_ic if std_ic != 0 else np.nan
    ic_win_rate = (ic_df['IC'] > 0).mean()

    # 顯著IC勝率 (p < 0.05)
    sig_ic_win_rate = (ic_df['p_value'] < 0.05).mean()

    # 自相关性 (滞后1阶)
    autocorr = ic_df['IC'].autocorr(lag=1)

    metrics = {
        '平均IC': avg_ic,
        'IC标准差': std_ic,
        'ICIR': icir,
        'IC胜率': ic_win_rate,
        '显著IC胜率(p<0.05)': sig_ic_win_rate,
        '自相关性': autocorr
    }

    return metrics, ic_df


def main():
    try:
        print("--- 正在讀取數據 ---")
        # 讀取alpha101.csv
        alpha_data = pd.read_csv('alpha101.csv')
        alpha_data['date'] = pd.to_datetime(alpha_data['date'])

        # 讀取原始數據以計算次日回報
        original_data = pd.read_csv('CI指数_日线数据.csv')
        original_data['date'] = pd.to_datetime(original_data['date'])
        original_data.rename(columns={'order_book_id': 'idx'}, inplace=True)

        # 創建一個新的DataFrame來存儲次日回報
        next_return_df = original_data.copy()
        next_return_df['next_return'] = next_return_df.groupby('idx')['close'].pct_change().shift(-1)
        next_return_series = next_return_df[['date', 'idx', 'next_return']]

        # 找到所有的alpha因子列
        alpha_columns = [col for col in alpha_data.columns if col.startswith('alpha')]
        if not alpha_columns:
            print("錯誤：在 alpha101.csv 中沒有找到任何以 'alpha' 開頭的因子列。")
            return

        print(f"找到以下因子進行測試: {alpha_columns}")

        all_metrics = {}

        for alpha_col in alpha_columns:
            if alpha_col=='alpha061' or alpha_col=='alpha075' or alpha_col=='alpha095':
                continue
            print(f"\n--- 正在測試因子: {alpha_col} ---")

            # 對因子值進行預處理 (去極值和標準化)
            temp_df = alpha_data[['date', 'idx', alpha_col]].copy()
            temp_df.rename(columns={alpha_col: 'alpha_value'}, inplace=True)
            temp_df['alpha_value'] = temp_df.groupby('date')['alpha_value'].transform(winsorize_standardize)

            metrics, _ = calculate_ic_metrics(temp_df, next_return_series)
            if metrics:
                all_metrics[alpha_col] = metrics

        # 將結果轉換為DataFrame並轉置
        if not all_metrics:
            print("\n未能計算任何因子的性能指標。")
            return

        results_df = pd.DataFrame(all_metrics)

        # 儲存為CSV文件
        output_file_path = 'alpha_performance_metrics.csv'
        results_df.to_csv(output_file_path, encoding='utf-8-sig')

        print(f"\n所有因子的性能指標已成功儲存至 {output_file_path}")
        print("以下是最終的結果表格:")
        print(results_df)

    except FileNotFoundError as e:
        print(f"錯誤：找不到所需的檔案。請確認 'alpha101.csv' 和 'CI指數_日线数据.csv' 是否在正確的路徑下。")
    except Exception as e:
        print(f"執行過程中發生未知錯誤：{e}")


if __name__ == "__main__":
    main()
