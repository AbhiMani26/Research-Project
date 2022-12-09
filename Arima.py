from math import sqrt

import pandas
from matplotlib import pyplot
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


def arima_rmse():
    series = read_csv('n225.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    series1 = read_csv('nasdaq.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    series2 = read_csv('hsi.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    series3 = read_csv('gspc.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    series4 = read_csv('djimonthly.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    series5 = read_csv('djiweekly.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    series6 = read_csv('medicare.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    series7 = read_csv('housing.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    series8 = read_csv('tradeusdollor.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    series9 = read_csv('foodandbev.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    series10 = read_csv('ms.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    series11 = read_csv('transportation.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    arima_list = list()
    arima_list.append(rmseCalculation(series))
    arima_list.append(rmseCalculation(series1))
    arima_list.append(rmseCalculation(series2))
    arima_list.append(rmseCalculation(series3))
    arima_list.append(rmseCalculation(series4))
    arima_list.append(rmseCalculation(series5))
    arima_list.append(rmseCalculation(series6))
    arima_list.append(rmseCalculation(series7))
    arima_list.append(rmseCalculation(series8))
    arima_list.append(rmseCalculation(series9))
    arima_list.append(rmseCalculation(series10))
    arima_list.append(rmseCalculation(series11))
    arima_list
    name_list = ["n225","nasdaq","hsi","gspc","djimonthly","djiweekly","medicare","housing","tradeusdollor","foodandbev","ms","transportation"]

    rsmeDf = pandas.DataFrame({
        'stock name': name_list,
        'rsmeArima': arima_list
    })
    rsmeDf.to_csv('arimarsme.csv')



def rmseCalculation(series):
    # split into train and test sets
    X = series.values
    size = int(len(X) * 0.70)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
    #pyplot.plot(test)
    #pyplot.plot(predictions, color='red')
    #pyplot.show()
    return rmse


class Arima:
    # load dataset

    pass
