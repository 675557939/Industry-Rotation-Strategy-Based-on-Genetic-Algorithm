# 行业轮动策略

Produced by ZIA-Quantitative Investment Department

# 数据说明

数据区间： from 2011-01-01 to 2025-08-17

字段：order_book_id, date, open, close, high, low, volume, total_turnover

格式：csv

# 策略说明
主程序在'A-share-Industry-Rotation-Strategy-main\GP'

开源证券粗复现主要参考开源证券研报

遗传算法单次训练：将数据前80%的时间数据用于训练，剩下20%用于验证

遗传算法滚动训练：每个月重新训练一次，训练窗口为前300个交易日数据
