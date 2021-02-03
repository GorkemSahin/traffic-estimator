import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from io import StringIO

class DateModel():
    year: int
    month: int
    day: int

def stringify_params():
  return 'epoch' + str(num_epochs) + ',LR' + str(learning_rate) + ',input' + str(input_size) + ',hidden' + str(hidden_size) + ',layer' + str(num_layers) 

def scaling_window(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.2,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        #h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(ula[:,-1:])
        
        return out


def main(file,  num_epochs, learning_rate, num_layers, hidden_size, seq_length, start, test_start, end):
    data_csv = StringIO(file)
    input_size = 1
    num_classes = 1

    series = pd.read_csv(data_csv, sep=",")
    print("------------------------------------------------")
    #print(series)
    print("------------------------------------------------")
    #series = pd.read_csv('data/Sep_hourly.csv')
    start_date = datetime(start.year, start.month, start.day)
    test_start_date = datetime(test_start.year, test_start.month, test_start.day)
    end_date = datetime(end.year, end.month, end.day)
    prediction_series = series[(pd.to_datetime(series["date_time"]) >= test_start_date) & (pd.to_datetime(series["date_time"]) < end_date)]
    print(prediction_series)
    series = series[(pd.to_datetime(series["date_time"]) >= start_date) & (pd.to_datetime(series["date_time"]) < end_date)]
    #print(training_set.head())
    #print(training_set.describe())

    training_set = series.iloc[:,1:2].values
    # print("------------------------------------------------")
    # print(training_set)
    # print(training_set.head())
    # print(training_set.describe())

    #plt.plot(training_set, label = 'Data')
    #plt.show()



    # Normalizing the data points to 0-1
    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)



    x, y = scaling_window(training_data, seq_length)

    # print(x)
    # print(y)
    # input()

    #train_size = int(len(y) * 0.8)
    train_size = len(training_set) - len(prediction_series) - seq_length - 1
    print(train_size)

    #test_size = len(prediction_series) -

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))


    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        outputs = lstm(trainX)
        
        # obtain the loss function
        loss = criterion(outputs.squeeze(1), trainY)
        
        loss.backward()
        
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    lstm.eval()
    train_predict = lstm(testX)
    train_predict = train_predict.squeeze(1)

    #print(train_predict)

    data_predict = train_predict.data.numpy()
    dataY_plot = testY.data.numpy()
    #print(data_predict)
    data_predict = sc.inverse_transform(data_predict)
    lst2 = [item[0] for item in data_predict]
    prediction_series["size"] = lst2
    dataY_plot = sc.inverse_transform(dataY_plot)
    # dataTMP = [*np.zeros(553), *data_predict]

    plot_directory = "/Users/gbot/Desktop/thesis/lstm_prediction/"
    dataTMP = data_predict
    residuals = dataY_plot - data_predict
    mape = round(np.mean(abs(residuals/dataY_plot)),4)
    #print(mape)
    predicted_series = pd.Series(data=prediction_series["size"].values, name="size", index=pd.to_datetime(prediction_series["date_time"]))
    series = pd.Series(data=series["size"].values, name="size", index=pd.to_datetime(series["date_time"]))
    plt.figure(figsize=(18,8))
    plt.legend(('Data', 'Predictions'), fontsize=16)
    plt.title('Mean Absolute Percent Error: ' + str(mape), fontsize=20)
    plt.ylabel('Traffic Per Hour', fontsize=16)
    plt.plot(series)
    plt.plot(predicted_series)
    # plt.plot(dataY_plot)
    # plt.plot(dataTMP)
    plt.suptitle('Time-Series Prediction')
    #plt.savefig(plot_directory + stringify_params() + '.png', bbox_inches='tight', dpi=500)
    #series.to_json(plot_directory + "json/series.json", date_format="iso")
    #predicted_series.to_json(plot_directory + "json/predicted_series.json", date_format="iso")
    pred = predicted_series.to_json()
    full_data = series.to_json()
    
    return([pred, full_data, mape])
    #plt.show()


# start = DateModel(2020, 9, 1)
# test_start = DateModel()
# end = DateModel()

# main("a",
#     num_epochs = 500,
#     learning_rate = 0.001,
#     num_layers = 2,
#     hidden_size = 16,
#     seq_length = 8,
#     )