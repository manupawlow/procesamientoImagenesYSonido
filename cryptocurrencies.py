from cProfile import label
import matplotlib.pyplot as plt
import pandas_datareader as web
import mplfinance as mpf
from pyparsing import col
import seaborn as sns
import datetime as dt
import numpy as np
import pandas as pd

currency = 'USD'
metric = 'Close'

startDate = '2018-4-1'
endDate   = '2022-4-1'

targetStock = "ZM" #["FB", "TSLA", "GOOG", "AMZN", "GLOB", "ACN"]

crypto = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'BNB-USD', 'SOL-USD', 'MATIC-USD', 'DOGE-USD']
# crypto = crypto[0:4]

COLORS = {
    "FB": "#BAD02F",
    "TSLA": "#BAD02F",
    "GOOG": "#BAD02F",
    "GLOB": "#BAD02F",
    "ACN": "#BAD02F",
    "ZM": "#5098F6",

    "BTC-USD": "#EF8E19",
    "ETH-USD": "#898989",
    "LTC-USD": "#325A98",
    "BNB-USD": "#EBB42F",
    "SOL-USD": "#62F09A",
    "MATIC-USD": "#7E45DE",
    "DOGE-USD": "#B59A2E",

    "MA7": "#FF0000",
    "MA20": "#00FF00",
    "MA200": "#0000FF",
}

# RETRIEVE DATA
stockData = web.DataReader(targetStock, "yahoo", startDate, endDate)
combined = stockData[[metric]].copy()
colnames = [targetStock]
combined.columns = colnames

for ticker in crypto:
    data = web.DataReader(ticker, "yahoo", startDate, endDate)
    combined = combined.join(data[metric])
    colnames.append(ticker)
    combined.columns = colnames

# SHOW GRAPHS
plt.yscale('log')

plt.plot(combined[targetStock], label=targetStock, color=COLORS[targetStock])
for ticker in crypto:
    plt.plot(combined[ticker], label=ticker, color=COLORS[ticker])
plt.legend(loc="upper right")
plt.figure()

# CORRELATION HEATMAP
correlations = combined.pct_change().corr(method="pearson")
sns.heatmap(correlations, annot=True, cmap="coolwarm")
plt.figure()

selectedCrypto = correlations[targetStock].sort_values(ascending=False).index[1]

print("Most correlational crypto: " + selectedCrypto)

# MOVING AVERAGE

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

ma7 = moving_average(combined[selectedCrypto], 7)
ma20 = moving_average(combined[selectedCrypto], 20)
ma200 = moving_average(combined[selectedCrypto], 200)

plt.plot(combined[selectedCrypto], label=selectedCrypto, color=COLORS[selectedCrypto])
plt.plot(combined[selectedCrypto].index, ma7, alpha=0.6, label="MA7", color=COLORS["MA7"])
plt.plot(combined[selectedCrypto].index, ma20, alpha=0.5, label="MA20", color=COLORS["MA20"])
plt.plot(combined[selectedCrypto].index, ma200, alpha=0.4, label="MA200", color=COLORS["MA200"])

plt.plot(combined[selectedCrypto].ewm(span=7).mean(), alpha=0.9, label="MA7(2)", color='r')


plt.legend(loc="upper right")

# RSI
delta = combined[selectedCrypto].diff(1)
delta.dropna(inplace=True)

positive = delta.copy()
negative = delta.copy()

positive[positive < 0] = 0
negative[negative > 0] = 0

days = 14

average_gain = positive.rolling(window=days).mean()
average_loss = abs(negative.rolling(window=days).mean())

relative_strenght = average_gain / average_loss
RSI = 100.0 - (100.0 / (1.0 + relative_strenght))

combinedRSI = pd.DataFrame()
combinedRSI['Adj Close'] = combined[selectedCrypto]
combinedRSI['RSI'] = RSI

plt.figure(figsize=(12,8))
ax1 = plt.subplot(211)
ax1.plot(combinedRSI.index, combinedRSI['Adj Close'], color=COLORS[selectedCrypto])
ax1.set_title("Close Price", color='white')

ax1.grid(True, color='#555555')
ax1.set_axisbelow(True)
ax1.set_facecolor('black')
ax1.figure.set_facecolor('#121212')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(combinedRSI.index, combinedRSI['RSI'], color='lightgray')
ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')

ax2.set_title("RSI Value", color='white')
ax2.grid(False)
ax2.set_axisbelow(True)
ax2.set_facecolor('black')
ax2. tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')

plt.show()

