# evaluate mlp
import pandas as pd
from math import sqrt
from numpy import array
from numpy import mean
from numpy import std
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from matplotlib import pyplot
from datetime import datetime
from datetime import timedelta
from time import time
import matplotlib.pyplot as plt
from io import StringIO

# define model parameters
neuron_count = 256
dropout_rate = 0.2
learning_rate = 0.0008
err = 0.0
# transform list into supervised learning format
def series_to_supervised(data, n_in=1, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# fit a model
def model_fit(train, val, config):
	# unpack config
	n_input, n_nodes, n_epochs, n_batch = config
	# prepare data
	data = series_to_supervised(train, n_in=n_input)
	train_x, train_y = data[:, :-1], data[:, -1]
	val_data = series_to_supervised(val, n_in=n_input)
	val_x, val_y = val_data[:, :-1], val_data[:, -1]

	model = Sequential()
	model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
	model.add(Dropout(dropout_rate))
	model.add(Dense(neuron_count, activation='relu'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1))
	optimizer = Adam(learning_rate=learning_rate)
	model.compile(loss='mse', optimizer=optimizer)
	model.summary()
	hist = model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, validation_data=(val_x, val_y), verbose=2)
	return model, hist

# forecast with a pre-fit model
def model_predict(model, history, config):
	# unpack config
	n_input, _, _, _ = config
	# prepare data
	x_input = array(history[-n_input:]).reshape(1, n_input)
	# forecast
	yhat = model.predict(x_input, verbose=0)
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(train, test, cfg):
	predictions = list()
	# fit model
	model, hist = model_fit(train, test, cfg)

	# plt.plot(hist.history['loss'])
	# plt.plot(hist.history['val_loss'])
	# plt.title('model train and validation loss')
	# plt.ylabel('loss')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'val'], loc='upper left')
	# plt.show()

	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return predictions, error

# repeat evaluation of a config
def repeat_evaluate(training_series, testing_series, config, n_repeats=30):
	# fit and evaluate the model n times
	predictions, scores = walk_forward_validation(training_series, testing_series, config)
	scores = [scores for _ in range(n_repeats)]
	return predictions, scores

# summarize model performance
def summarize_scores(name, scores):
	# print a summary
	scores_m, score_std = mean(scores), std(scores)
	print('%.3f MAPE (+/- %.3f)' % (scores_m, score_std))
	# box and whisker plot
	# pyplot.boxplot(scores)
	# pyplot.show()
	global err
	err = scores_m
	return scores_m

def main(file, epoch, neuron, learning, dropout, start, test_start, end):
	global neuron_count
	global dropout_rate
	global learning_rate
	global err
	neuron_count = neuron
	dropout_rate = dropout
	learning_rate = learning

	data_csv = StringIO(file)
	#df = pd.read_csv(data_csv, sep=",")

	scaler = StandardScaler()
	df = pd.read_csv("data/Sep_hourly.csv")
	df['size_norm'] = scaler.fit_transform(df['size'].values.reshape(-1,1))
	series = pd.Series(data=df["size_norm"].values, index=pd.to_datetime(df["date_time"]))
	start_date = datetime(start.year, start.month, start.day)
	test_start_date = datetime(test_start.year, test_start.month, test_start.day)
	end_date = datetime(end.year, end.month, end.day)
	training_series = series[start_date:test_start_date].values
	testing_series = series[test_start_date:end_date].values
	# define config
	config = [24, 128, epoch, 100]
	# grid search
	predictions, scores = repeat_evaluate(training_series, testing_series, config)
	predicted_series = series[test_start_date:end_date]
	predicted_series = pd.Series(data=[item[0] for item in predictions], name="size", index=pd.to_datetime(predicted_series.index))
	print(predicted_series)
	
	series =  pd.Series(data=scaler.inverse_transform(series.values), name="size", index=pd.to_datetime(series.index))
	predicted_series = pd.Series(data=scaler.inverse_transform(predicted_series.values), name="size", index=pd.to_datetime(predicted_series.index))

	mape = summarize_scores('mlp', scores)
	print("-----------------------------")
	print(err)
	print("-----------------------------")
	pred = predicted_series.to_json()
	full_data = series.to_json()
	return([pred, full_data, err])

#main("a", 65, 128, 0.2, 0.0008)