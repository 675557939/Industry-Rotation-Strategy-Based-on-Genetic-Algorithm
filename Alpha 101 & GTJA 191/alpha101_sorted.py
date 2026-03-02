import pandas as pd

df = pd.read_csv('alpha_performance_metrics.csv', index_col=0).transpose()

df['abs_ICIR'] = df['ICIR'].abs()

df['rank_ICIR'] = df['abs_ICIR'].rank(ascending=False)
df['rank_IC胜率'] = df['IC胜率'].rank(ascending=False)
df['rank_显著IC胜率'] = df['显著IC胜率(p<0.05)'].rank(ascending=False)
df['rank_自相关性'] = df['自相关性'].rank(ascending=True)

weights = {'rank_ICIR': 0.5, 'rank_显著IC胜率': 0.3, 'rank_IC胜率': 0.1, 'rank_自相关性': 0.1}
df['综合得分'] = sum(df[k] * w for k, w in weights.items())

df_sorted = df.sort_values('综合得分')
result = df_sorted[['平均IC', 'IC标准差', 'ICIR', 'IC胜率', '显著IC胜率(p<0.05)', '自相关性', '综合得分']]
result['排名'] = range(1, len(result) + 1)
result.to_csv('alpha101_sorted.csv', index_label='Alpha')
