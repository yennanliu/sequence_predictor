# python 3 


"""
# predict time-series-data with tensorflow RNN model 
# credit 
# https://github.com/JustinBurg/TensorFlow_TimeSeries_RNN_MapR/blob/master/RNN_Timeseries_Demo.ipynb


"""


# import library 
import pandas as pd 
import numpy as np 
import tensorflow as tf
from tensorflow.contrib import rnn

# import data 
from data_prepare import *



df_fb = load_data('FB')
col = ['Date', ' Open', ' High', ' Low', ' Close', ' Volume']
df_fb = df_fb[col]
df_fb.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
