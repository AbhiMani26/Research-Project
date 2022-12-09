# This is a sample Python script.
from Arima import Arima, arima_rmse
from LstmModel import LSTM, lstm_rmse, LstmModel

def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arima = Arima()
    arima_rmse()
    lstm = LstmModel()
    lstm_rmse()
