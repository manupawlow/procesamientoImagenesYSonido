import pandas_datareader as web

from config import *

# RETRIEVE DATA
stockData = web.DataReader(targetStock, "yahoo", startDate, endDate)
combined = stockData[[metric]].copy()
colnames = [targetStock]
combined.columns = colnames

for ticker in crypto:
    print('Reading ' + ticker + '...')
    data = web.DataReader(ticker, "yahoo", startDate, endDate)
    combined = combined.join(data[metric])
    colnames.append(ticker)
    combined.columns = colnames