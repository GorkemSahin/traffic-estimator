import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
register_matplotlib_converters()
from time import time
from io import StringIO

err = ""

def stringify_params_errors():
  return '_' + str(p) + ',' + str(d) + ',' + str(q) + ',' + str(P) + ',' + str(D) + ',' + str(Q) + ',' + str(s)

def train(data, order, seasonal_order, print_summary=True):
  model = SARIMAX(data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
  start = time()
  model_fit = model.fit(maxiter=999)
  end = time()
  print('Model Fitting Time:', end - start)
  if print_summary:
    print(model_fit.summary())
  return model_fit

def predict(series, model_fit, test_data, display_residuals=True, display_predictions=True):
  global err
  predictions = model_fit.forecast(len(test_data))
  predictions = pd.Series(predictions, index=test_data.index)
  residuals = test_data - predictions
  err = str(round(np.mean(abs(residuals/test_data)),4))
  print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/test_data)),4))
  print('Root Mean Squared Error:', np.sqrt(np.mean(residuals**2)))
  if display_residuals:
    plt.figure(figsize=(18,8))
    plt.plot(residuals)
    plt.axhline(0, linestyle='--', color='k')
    plt.title('Residuals from SARIMA Model', fontsize=20)
    plt.ylabel('Error', fontsize=16)
    #plt.savefig(plot_directory + month + '_' + stringify_params_errors() + '_residuals.png', bbox_inches='tight', dpi=500)
    #plt.show()
  if display_predictions:
    plt.figure(figsize=(18,8))
    plt.plot(series)
    plt.plot(predictions)
    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.title('Mean Absolute Percent Error: ' + str(err), fontsize=20)
    plt.ylabel('Traffic Per Hour', fontsize=16)
    #plt.savefig(plot_directory + month + '_' + stringify_params_errors() + '.png', bbox_inches='tight', dpi=500)
    #plt.show()
  return predictions, residuals

def main(file, p, d, q, P, D, Q, s, start, test_start, end):
  #month = "Sep"
  plot_directory = "/Users/gbot/Desktop/thesis/plot/prediction/"
  # 0 0 0 8 0 7 24
  # p = 0
  # d = 0
  # q = 0
  # P = 8
  # D = 0
  # Q = 7
  # s = 24


  data_csv = StringIO(file)
  df = pd.read_csv(data_csv, sep=",")
  series = pd.Series(data=df["size"].values, index=pd.to_datetime(df["date_time"]))

  start_date = datetime(start.year, start.month, start.day)
  test_start_date = datetime(test_start.year, test_start.month, test_start.day)
  end_date = datetime(end.year, end.month, end.day)

  series = series[start_date:end_date]
  train_data = series[start_date:test_start_date]
  test_data = series[test_start_date:end_date]

  model_fit = train(data=train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
  predictions, residuals = predict(series, model_fit, test_data)

  # predictions.to_json(plot_directory + month + "_predictions.json", date_format="iso")
  # residuals.to_json(plot_directory + month + "_residuals.json", date_format="iso")
  # print(predictions)
  # print(residuals)

  pred = predictions[1:].to_json()
  full_data = series.to_json()
  print("-----------------------------")
  print(err)
  print("-----------------------------")
  return([pred, full_data, err])

#main()