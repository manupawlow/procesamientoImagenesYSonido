from turtle import color

from pyparsing import java_style_comment
import pandas_datareader as web
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


ticker = 'FB'
#cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'LTC-USD', 'SOL-USD', 'MATIC-USD', 'DOGE-USD']
#colors =  ['b', '#EF8E19', '#888888', '#EBB42F', '#325A98', '#64F496', '#7E45DE'  , '#B59A30']

rows = 1
cols = 2
tickers = [
    [       'FB',  'BTC-USD'],
    # [  'ETH-USD',  'BNB-USD'],
    # [  'LTC-USD',  'SOL-USD'],
    # ['MATIC-USD', 'DOGE-USD']
]

colors = [
    ['#0000FF', '#EF8E19'],
    ['#888888', '#EBB42F'],
    ['#325A98', '#64F496'],
    ['#7E45DE', '#B59A30']
]

startDate = '2021-4-1' #dt.datetime(2018, 1, 1)
endDate   = '2022-4-1' #dt.datetime.now()

fig, axs = plt.subplots(rows, cols)

closes = []

for i in range(len(tickers)):
  for j in range(len(tickers[i])):
    data = web.DataReader(tickers[i][j], 'yahoo', startDate, endDate)
    closes.append(data)
    # axs[i, j].plot(data.index, data['Adj Close'], color=colors[i][j])
    # axs[i, j].set_title(tickers[i][j], x=0.05, y=0.8)

# plt.show()

print(closes.corr())
sns.heatmap(closes)
