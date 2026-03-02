import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("alpha101.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["y"] = (df["close"].shift(-1) - df["close"]) / df["close"]
df["y"] = df["y"].replace([np.inf, -np.inf], np.nan)

alpha_cols = [c for c in df.columns if c.lower().startswith("alpha")]
for col in alpha_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
df[alpha_cols] = df[alpha_cols].clip(-1e36, 1e36)

model_df = df[alpha_cols + ["y"]].dropna()
X, y = model_df[alpha_cols], model_df["y"]

n = len(model_df)
train_end, valid_end = int(n * 0.7), int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_valid, y_valid = X.iloc[train_end:valid_end], y.iloc[train_end:valid_end]
X_test, y_test = X.iloc[valid_end:], y.iloc[valid_end:]

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

for name, X_eval, y_eval in [("验证集", X_valid, y_valid), ("测试集", X_test, y_test)]:
    pred = rf.predict(X_eval)
    rmse = np.sqrt(mean_squared_error(y_eval, pred))
    r2 = r2_score(y_eval, pred)
    print(f"{name}: RMSE={rmse:.6f}, R2={r2:.4f}")

y_pred = rf.predict(X_test)
direction_acc = ((y_test > 0).astype(int) == (y_pred > 0).astype(int)).mean()
print(f"涨跌方向准确率: {direction_acc:.4f}")

importances = pd.Series(rf.feature_importances_, index=alpha_cols).sort_values(ascending=False)
print("\nTop 20 特征:")
print(importances.head(20))
