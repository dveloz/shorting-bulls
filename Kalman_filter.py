# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 22:55:10 2021

@author: Arquimedes
"""

##Filtro de Kalman 

from pykalman import KalmanFilter
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError
import datetime

start_date = '2021-01-01'
end_date = str(datetime.datetime.now().strftime('%Y-%m-%d'))

ticker = input('Introduzca su ticker: ')



def get_data(ticker):
    try:
        stock_data = data.DataReader(ticker, 'yahoo', start_date, end_date)
    except RemoteDataError:
        print('No data found for {t}'.format(t=ticker))
    return stock_data


stock = get_data(ticker)
x = stock.Close


#Construccion de kalman filter

kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance = 1,
                  transition_covariance=.05)

state_means, _= kf.filter(x.values)
state_means = pd.Series(state_means.flatten(), index = stock.index)


mean10 = x.rolling(window = 10).mean()
mean30 = x.rolling(window = 30).mean()
mean60 = x.rolling(window = 60).mean()
mean90 = x.rolling(window = 90).mean()





#Plot original para vizualizar mejor 

plt.plot(state_means[-100:])
plt.plot(x[-100:])
plt.plot(mean30[-100:])
plt.plot(mean60[-100:])
plt.plot(mean90[-100:])
plt.title('Kalman filter estimate of average')
plt.legend(['Kalman estimate','Precio','30-day Moving average','60 day Moving average','90-day Moving average'])
plt.xlabel('Day')
plt.ylabel('Price');