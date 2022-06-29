from cProfile import label
from copyreg import constructor
from typing import OrderedDict
import matplotlib.pyplot as plt
import mplfinance as mpf
from pyparsing import col
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pmdarima
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from  statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn import metrics
from sklearn.metrics import mean_squared_error

import seaborn as sns

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from config import *
from data import combined

# SHOW GRAPHS
ax = plt.subplot(1,1,1)
ax.set_title(targetStock + " and Cryptocurrencies Close Price", color='white')

ax.grid(True, color='#555555')
ax.set_axisbelow(True)
ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

plt.yscale('log')

plt.plot(combined[targetStock], label=targetStock, color=COLORS[targetStock])
for ticker in crypto:
    plt.plot(combined[ticker], label=ticker, color=COLORS[ticker], alpha=0.6)

plt.legend(loc="upper right")
plt.figure()

# CORRELATION HEATMAP
correlations = combined.pct_change().corr(method="pearson")
selectedCrypto = correlations[targetStock].sort_values(ascending=False).index[1]

sns.heatmap(correlations, annot=True, cmap="coolwarm")
plt.figure()

print("Most correlational crypto: " + selectedCrypto)

# MOVING AVERAGE
ax1 = plt.subplot(1,1,1)
ax1.set_title(selectedCrypto + " Close Price", color='white')

ax1.grid(True, color='#555555')
ax1.set_axisbelow(True)
ax1.set_facecolor('black')
ax1.figure.set_facecolor('#121212')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'same') / w
# ma7 = moving_average(combined[selectedCrypto], 7)
# ma20 = moving_average(combined[selectedCrypto], 20)
# ma200 = moving_average(combined[selectedCrypto], 200)

ma7 = combined[selectedCrypto].ewm(span=7).mean()
ma20 = combined[selectedCrypto].ewm(span=20).mean()
ma200 = combined[selectedCrypto].ewm(span=200).mean()

plt.plot(combined[selectedCrypto], alpha=0.95, label=selectedCrypto, color='white')
plt.plot(combined[selectedCrypto].index, ma7, alpha=0.8, label="MA7", color=COLORS["MA7"])
plt.plot(combined[selectedCrypto].index, ma20, alpha=0.8, label="MA20", color=COLORS["MA20"])
plt.plot(combined[selectedCrypto].index, ma200, alpha=0.8, label="MA200", color=COLORS["MA200"])

# plt.legend(loc="upper right")

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

# ARIMA: Test/Train Data
alpha = 0.9
df = combined[selectedCrypto]
to_row = int(len(df) * alpha)

training_data = list(df[0:to_row])
testing_data = list(df[to_row:])

plt.figure(figsize=(10,6))
ax = plt.subplot(1,1,1)
ax.set_title("Train/Test Data (90%/10%)", color='white')

ax.grid(True, color='#555555')
ax.set_axisbelow(True)
ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
# plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df[0:to_row], 'green', label='Train data')
plt.plot(df[to_row - 1:], 'blue', label='Test data')
plt.legend()

# ARIMA: Prediction
# model_autoARIMA = auto_arima(training_data, start_p=0, start_q=0,
#                     #   test='adf',       # use adftest to find optimal 'd'
#                     #   max_p=4, max_q=3, # maximum p and q
#                     #   m=1,              # frequency of series
#                     #   d=None,           # let model determine 'd'
#                     #   seasonal=False,   # No Seasonality
#                     #   start_P=0, 
#                     #   D=0, 
#                     #   trace=True,
#                     #   error_action='ignore',  
#                     #   suppress_warnings=True, 
#                     #   stepwise=True
#                     )
# # print(model_autoARIMA.summary())
# model_autoARIMA.plot_diagnostics()

# base_model = SARIMAX(training_data, order=(4,1,0)).fit()
# forecast = base_model.get_forecast(steps=len(testing_data))
# values = forecast.conf_int()

# pred_df = pd.DataFrame(columns = ['lower', 'pred', 'upper'])
# pred_df['pred'] = forecast.predicted_mean
# pred_df['lower'] = list(map(lambda x: x[0], values))
# pred_df['upper'] = list(map(lambda x: x[1], values))

# date_range = df[to_row:].index
# pred_df.index = date_range

# print(pred_df)

# # # Calculate MSE 
# # mse = mean_squared_error(pred, test)

# fig,ax = plt.subplots(figsize=(12,7))
# kws = dict(marker='o')

# # ax.plot(date_range, training_data,label='Train',**kws)
# ax.plot(date_range, testing_data,label='Test',**kws)
# ax.plot(pred_df['pred'],label='Prediction',ls='--',linewidth=3)

# ax.fill_between(x=pred_df.index,y1=pred_df['lower'],y2=pred_df['upper'],alpha=0.3)
# ax.set_title('Model Validation', fontsize=22)
# ax.legend(loc='upper left')
# fig.tight_layout()

model_predictions = []
n_test_obser = len(testing_data)
for i in range(n_test_obser):
    model = ARIMA(training_data, order=(4,1,0))
    model_fit = model.fit()
    output = model_fit.forecast(alpha=0.05)
    model_predictions.append(output[0])
    training_data.append(testing_data[i])

print(model_fit.summary())
print(model_fit.forecast(alpha=0.05))
date_range = df[to_row:].index

print(len(date_range))
print(len(model_predictions))
print(len(testing_data))

plt.figure(figsize=(15,9))
ax = plt.subplot(1,1,1)
ax.set_title("Train/Test Data", color='white')

ax.grid(True, color='#555555')
ax.set_axisbelow(True)
ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
# plt.grid(True)
plt.plot(date_range, model_predictions, color='red', alpha=0.6, marker='o', linestyle='dashed', label=selectedCrypto + ' predicted price')
plt.plot(date_range, testing_data, color=COLORS[selectedCrypto], label=selectedCrypto + ' actual price')
plt.title('Test data vs Prediction')
plt.xlabel('Dates')
plt.ylabel('Closing Prices')

# fc, se, conf = model_fit.forecast(15, alpha=0.05)  # 95% conf

# fc_series = pd.Series(fc, index=df[to_row:].index)
# lower_series = pd.Series(conf[:, 0], index=df[to_row:].index)
# upper_series = pd.Series(conf[:, 1], index=df[to_row:].index)
# plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)

# plt.legend()

# mape = np.mean(np.abs(np.array(model_predictions) - np.array(testing_data)) / np.abs(testing_data))
# print('Mean Absolute Percentage Error: ' + str(mape))



prediction_days = 5
future_predictions = model_fit.forecast(prediction_days, alpha=0.05)

# print(df)
# print('Predicted future price: ' + future_predictions[0])

# all_dates = [*(list(df.index)), *pd.date_range(endDate, periods=prediction_days).tolist()]
future_dates = pd.date_range(endDate, periods=prediction_days)

plt.plot(future_dates,future_predictions, marker='o', label='Future Price')

# plt.plot(all_dates, [*(list(df)), *pepe], label='PEPE')

# plt.figure(figsize=(10,6))
# ax = plt.subplot(1,1,1)
# ax.set_title("Train/Test Data", color='white')

# ax.grid(True, color='#555555')
# ax.set_axisbelow(True)
# ax.set_facecolor('black')
# ax.figure.set_facecolor('#121212')
# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')
# # plt.grid(True)
# plt.xlabel('Dates')
# plt.ylabel('Closing Prices')
# plt.plot(all_dates, df.values, 'green', label='Data')
# plt.plot(all_dates, future_predictions, 'blue', label='Prediction')
# plt.legend()

plt.show()
