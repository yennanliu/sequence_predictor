# python 3 

# ops 
import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
%pylab inline

# ML
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.callbacks import Callback


# UDF 
from model_LSTM_keras import * 
from data_prepare import *




def tune_Keras_LSTM(dataset):
    df = get_data(dataset)
    df_ = col_fix(df)
    dataset = get_train_data(df_)
    for k in range(1,5):
        print (k)
        dataset,trainPredict,testPredict,trainPredictPlot,testPredictPlot = one_input_LSTM_model_1_(dataset,lookback=k)
        plt.axvline(x=1050, color='r', linestyle='--')
        #plt.title('predict with lookback = {}'.format(k))
        if k > 1 :
            plt.plot(trainPredictPlot*187)
            plt.plot(testPredictPlot*187)
            plt.plot( df_[['Open']])
            plt.legend(['train-test-split','train_Predict', 'test_Predict','whole_dataset'])
            plt.title('Stock Price Prdict with LSTM  (lookback = {})'.format(k))
            plt.xlabel('timestamp')
            plt.ylabel('stock index')
            plt.show()
        else:
            plt.plot(trainPredictPlot)
            plt.plot(testPredictPlot)
            plt.plot( df_[['Open']])
            plt.legend(['train-test-split','train_Predict', 'test_Predict','whole_dataset'])
            plt.title('Stock Price Prdict with LSTM  (lookback = {})'.format(k))
            plt.xlabel('timestamp')
            plt.ylabel('stock index')
            plt.show()


