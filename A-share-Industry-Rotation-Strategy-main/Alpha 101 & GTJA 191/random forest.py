import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取数据
df = pd.read_csv("alpha101.csv")

# 把 date 转成日期类型，并按时间排序
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 构造预测目标 y
df["close_next"] = df["close"].shift(-1)
df["y"] = (df["close_next"] - df["close"]) / df["close"]
df["y"] = df["y"].replace([np.inf, -np.inf], np.nan)

# 找出 alpha 特征列
alpha_cols = [c for c in df.columns if c.lower().startswith("alpha")]
print("一共找到 alpha 特征数：", len(alpha_cols))

# 清洗 alpha 特征
for col in alpha_cols:

    # 转为数值（非数值变 NaN）
    df[col] = pd.to_numeric(df[col], errors="coerce")

    # 把 inf 换成 NaN
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# 裁剪极端值，防止float32溢出
df[alpha_cols] = df[alpha_cols].clip(lower=-1e36, upper=1e36)

# 组合建模数据并去掉缺失
model_df = df[alpha_cols + ["y"]].dropna()

X = model_df[alpha_cols]
y = model_df["y"]

print("可用于建模的样本数量：", len(model_df))

# 按时间划分训练 / 验证 / 测试集
n = len(model_df)

train_end = int(n * 0.7)
valid_end = int(n * 0.85)   # 70% train + 15% valid = 85%

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_valid = X.iloc[train_end:valid_end]
y_valid = y.iloc[train_end:valid_end]

X_test = X.iloc[valid_end:]
y_test = y.iloc[valid_end:]

print(f"训练集样本数：{len(X_train)}")
print(f"验证集样本数：{len(X_valid)}")
print(f"测试集样本数：{len(X_test)}")

# 训练随机森林
rf = RandomForestRegressor(
    n_estimators=300,        # 大森林
    max_depth=None,          # 不限制树深度
    min_samples_split=2,     # 非常容易分裂
    min_samples_leaf=1,      # 叶子节点很小
    max_features=None,       # 使用所有特征
)

rf.fit(X_train, y_train)

# 验证集表现（用于调参）
y_valid_pred = rf.predict(X_valid)

valid_mse = mean_squared_error(y_valid, y_valid_pred)
valid_rmse = np.sqrt(valid_mse)
valid_r2 = r2_score(y_valid, y_valid_pred)

print("\n===== 验证集表现 =====")
print(f"RMSE : {valid_rmse:.6f}")
print(f"R^2  : {valid_r2:.4f}")

# 测试集预测
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== 测试集回归指标 =====")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R^2  : {r2:.4f}")

# 涨跌预测准确率
direction_true = (y_test > 0).astype(int)
direction_pred = (y_pred > 0).astype(int)
direction_acc = (direction_true == direction_pred).mean()

print(f"\n涨跌方向预测准确率: {direction_acc:.4f}")

# 特征重要性
importances = pd.Series(rf.feature_importances_, index=alpha_cols)
importances = importances.sort_values(ascending=False)

print("\n===== 最重要的20个 alpha =====")
print(importances.head(20))
