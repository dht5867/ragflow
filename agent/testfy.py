import yfinance as yf
import logging

# 创建 Ticker 对象，表示对 Microsoft 公司的股票数据进行操作
msft = yf.Ticker("IBM")
# 获取所有的股票信息
logging.info(msft.info)
# 获取历史市场数据，这里是过去一个月的数据
hist = msft.history(period="1mo")

# 显示历史数据的元信息（需要先调用 history() 函数）
msft.history_metadata

# 显示公司行为信息（股利、拆股、资本收益）
msft.actions
msft.dividends
msft.splits
msft.capital_gains  # 仅适用于共同基金和交易所交易基金（etfs）

# 显示股票股数
msft.get_shares_full(start="2022-01-01", end=None)

# 显示财务报表：
# - 收入表
msft.income_stmt
msft.quarterly_income_stmt
# - 资产负债表
msft.balance_sheet
msft.quarterly_balance_sheet
# - 现金流量表
msft.cashflow
msft.quarterly_cashflow
# 若要查看更多选项，请参考 `Ticker.get_income_stmt()`

# 显示股东信息
msft.major_holders
msft.institutional_holders
msft.mutualfund_holders

# 显示未来和历史的盈利日期，返回最多未来4个季度和过去8个季度的数据，默认情况下。
# 注意：如果需要更多信息，可以使用 msft.get_earnings_dates(limit=XX)，其中 XX 为增加的限制参数。
msft.earnings_dates

# 显示国际证券识别码（ISIN） - *实验性功能*
# ISIN = International Securities Identification Number
msft.isin

# 显示期权到期日期
msft.options

# 显示新闻
msft.news

 