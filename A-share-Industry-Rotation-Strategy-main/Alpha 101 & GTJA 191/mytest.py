import pandas as pd
import numpy as np
from Alpha_code_1 import get_alpha
from pathlib import Path

# 確保文件路徑正確，如果檔案在當前目錄，則可簡化路徑
file_path = Path('CI指数_日线数据.csv')

# 定義需要處理的所有指數代碼
index_codes = [
    'CI005001.INDX', 'CI005002.INDX', 'CI005008.INDX', 'CI005014.INDX',
    'CI005019.INDX', 'CI005021.INDX', 'CI005024.INDX', 'CI005028.INDX',
    'CI005029.INDX', 'CI005030.INDX'
]

# 儲存所有指數的alpha因子的列表
all_alphas = []

try:
    # 從本地CSV文件讀取數據
    df = pd.read_csv(file_path)
    print("數據讀取成功。")

    # 補充計算 'amount', 'outstanding_share', 'turnover' 三個指標
    df['amount'] = df['close'] * df['volume']

    # 這裡假設流通股本為常數，請替換為實際數據以獲得更準確的結果
    df['outstanding_share'] = 100000000.0
    df['turnover'] = df['volume'] / df['outstanding_share']

    # 將 'date' 列轉換為日期時間格式
    df['date'] = pd.to_datetime(df['date'])

    # 確保數據按指數代碼和日期排序
    df = df.sort_values(by=['order_book_id', 'date']).reset_index(drop=True)

    # 遍歷所有指數代碼
    for index_code in index_codes:
        print(f"\n--- 正在處理指數: {index_code} ---")

        # 篩選出當前指數的數據
        stock_df = df[df['order_book_id'] == index_code].copy()

        # 如果數據為空，則跳過
        if stock_df.empty:
            print(f"警告：找不到指數 {index_code} 的數據，已跳過。")
            continue

        # 設置日期為索引，並計算日回報
        stock_df = stock_df.set_index('date')
        stock_df['change'] = stock_df['close'].pct_change()

        # 刪除帶有NaN值的行，以處理pct_change的第一行
        stock_df.dropna(subset=['change'], inplace=True)

        # 計算Alpha因子
        try:
            alpha = get_alpha(stock_df)

            # 將指數代碼和日期都加入到alpha的DataFrame中
            alpha['idx'] = index_code
            alpha['date'] = alpha.index

            # 儲存結果
            all_alphas.append(alpha)

            print(f"指數 {index_code} 的 Alpha 因子計算成功。")

        except Exception as e:
            print(f"錯誤：計算指數 {index_code} 的 Alpha 因子時發生錯誤：{e}")

    # 檢查是否有任何因子被計算
    if not all_alphas:
        print("\n未能計算任何Alpha因子，請檢查數據來源和程式碼。")
    else:
        # 合併所有結果
        final_alpha_df = pd.concat(all_alphas, ignore_index=False)
        final_alpha_df = final_alpha_df.reset_index()

        # 儲存為CSV文件
        output_file_path = 'alpha101.csv'
        final_alpha_df.to_csv(output_file_path, index=False)

        print(f"\n所有指數的 Alpha 因子已成功合併並儲存至 {output_file_path}")
        print("以下是最終合併數據的前5行:")
        print(final_alpha_df.head())

except FileNotFoundError:
    print(f"錯誤：找不到文件 '{file_path}'。請確認文件路徑是否正確。")
except Exception as e:
    print(f"發生未知錯誤：{e}")
