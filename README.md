# 行业轮动策略

Produced by ZIA-Quantitative Investment Department

# 数据说明

数据：中信一级行业数据

数据区间： from 2015-01-05 to 2024-12-31

字段：order_book_id, date, open, close, high, low, volume, total_turnover

格式：csv

# 框架说明

主程序在A-share-Industry-Rotation-Strategy-main/GP/遗传算法滚动训练

开源证券粗复现.py and 遗传算法单次训练.py 作为对比试验，主要变量在于适应度函数、特征工程、算子工程、GP参数、因子训练方式（单次/多次）

alpha_analysis.py 用于分析因子质量

cl_factors.ipynb 为手动挖掘因子及其表现
