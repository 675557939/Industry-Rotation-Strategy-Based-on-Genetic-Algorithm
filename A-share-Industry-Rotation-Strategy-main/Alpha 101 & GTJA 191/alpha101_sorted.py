import pandas as pd

# 读取数据
df = pd.read_csv('alpha_performance_metrics.csv', index_col=0).transpose()  # 索引为alpha名称

# 计算绝对值
df['abs_ICIR'] = df['ICIR'].abs()
df['abs_mean_IC'] = df['平均IC'].abs()   # 可选，但不用

# 计算排名：
# 注意：排名方法：ascending=False为降序（值越大排名越小），ascending=True为升序（值越小排名越小）
df['rank_ICIR'] = df['abs_ICIR'].rank(ascending=False)   # 值越大排名越小（越好）
df['rank_IC胜率'] = df['IC胜率'].rank(ascending=False)
df['rank_显著IC胜率'] = df['显著IC胜率(p<0.05)'].rank(ascending=False)
df['rank_自相关性'] = df['自相关性'].rank(ascending=True)   # 自相关性升序：值越小排名越小

# 加权综合得分（权重）
weights = {'rank_ICIR': 0.5, 'rank_显著IC胜率': 0.3, 'rank_IC胜率': 0.1, 'rank_自相关性': 0.1}
df['综合得分'] = df['rank_ICIR'] * weights['rank_ICIR'] + df['rank_显著IC胜率'] * weights['rank_显著IC胜率'] + df['rank_IC胜率'] * weights['rank_IC胜率'] + df['rank_自相关性'] * weights['rank_自相关性']

# 按综合得分升序排序（得分越小排名越前）
df_sorted = df.sort_values(by='综合得分')

# 输出到alpha101_sorted.csv，包含alpha名称和排名（以及所需信息）
result = df_sorted[['平均IC', 'IC标准差', 'ICIR', 'IC胜率', '显著IC胜率(p<0.05)', '自相关性', '综合得分']]
result['排名'] = range(1, len(result)+1)   # 添加排名列
result.to_csv('alpha101_sorted.csv', index_label='Alpha')