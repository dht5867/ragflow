
import akshare as ak

# 获取贵州茅台日线数据

stock_zh_a_daily_df = ak.stock_zh_a_daily(
    symbol="sh600519",  # 贵州茅台股票代码
    adjust="qfq"         # 前复权处理
)
print(stock_zh_a_daily_df.head(100))


import akshare as ak
 
# 获取A股实时行情（基础版）
#stock_zh_a_spot_df = ak.stock_zh_a_spot()
#print(stock_zh_a_spot_df[['代码', '名称', '最新价', '涨跌幅']].head())