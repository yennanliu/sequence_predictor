
# python 3 

"""
credit 
https://github.com/thushv89/datacamp_tutorials/blob/master/Reviewed/lstm_stock_market_prediction.ipynb

"""


# ops 
import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime as dt

# ML 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


#--------------------------------------------------
# help functions 

def get_data(fine_name):
	print (fine_name)
	df = pd.read_csv('data/{}.csv'.format(fine_name))
	print (df.head())
	return df  

def col_fix(df):
    df = df.drop('Unnamed: 0', axis=1) 
    print (df.columns)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    #df['Date'] = pd.to_datetime(df['Date'] )
    print (df.head())
    return df 



def preprocess_data(df):
	# Breaking Data to Train and Test and Normalizing Data
	# First calculate the mid prices from the highest and lowest 
	high_prices = df.loc[:,'High'].as_matrix()
	low_prices = df.loc[:,'Low'].as_matrix()
	mid_prices = (high_prices+low_prices)/2.0
	# get train- test data 
	train_data = mid_prices[:1000]
	test_data = mid_prices[1000:]
	# Scale the data to be between 0 and 1
	# When scaling remember! You normalize both test and train data w.r.t training data
	# Because you are not supposed to have access to test data
	scaler = MinMaxScaler()
	train_data = train_data.reshape(-1,1)
	test_data = test_data.reshape(-1,1)
	return train_data, test_data, scaler


def plot_data(df):
	plt.figure(figsize = (18,9))
	plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
	plt.xticks(range(0,df.shape[0],30),df['Date'].loc[::30],rotation=45)
	plt.xlabel('Date',fontsize=18)
	plt.ylabel('Mid Price',fontsize=18)
	plt.show()


#--------------------------------------------------
# ML 

def model_avg_prev_scalar(train_data,test_data):
	# Train the Scaler with training data and smooth data 
	smoothing_window_size = 20
	for di in range(0,900,smoothing_window_size):
	    scaler.fit(train_data[di:di+smoothing_window_size,:])
	    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])
	#-------------------------
	# You normalize the last bit of remaining data 
	scaler.fit(train_data[di+smoothing_window_size:,:])
	train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:]) 
	# Reshape both train and test data
	train_data = train_data.reshape(-1)
	# Normalize test data
	test_data = scaler.transform(test_data).reshape(-1)
	#-------------------------
	# Now perform exponential moving average smoothing
	# So the data will have a smoother curve than the original ragged data
	EMA = 0.0
	gamma = 0.1
	for ti in range(1000):
	  EMA = gamma*train_data[ti] + (1-gamma)*EMA
	  train_data[ti] = EMA

	# Used for visualization and test purposes
	all_mid_data = np.concatenate([train_data,test_data],axis=0)
	window_size = 20
	N = train_data.size
	std_avg_predictions = []
	std_avg_x = []
	mse_errors = []

	for pred_idx in range(window_size,N):
	    
	    if pred_idx >= N:
	        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
	    else:
	        date = df_FB_.loc[pred_idx,'Date']
	        
	    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
	    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
	    std_avg_x.append(date)

	print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))
	# plot train results 
	plt.figure(figsize = (18,9))
	plt.plot(range(df_FB_.shape[0]),all_mid_data,color='b',label='True')
	plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
	#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
	plt.xlabel('Date')
	plt.ylabel('Mid Price')
	plt.legend(fontsize=18)
	plt.show()
	#return all_mid_data, 



#--------------------------------------------------


if __name__ == '__main__':
	df_FB = get_data('FB')
	df_FB_ = col_fix(df_FB)
	# re-order the datetime 
	# https://stackoverflow.com/questions/28161356/sort-pandas-dataframe-by-date
	df_FB_.Date= pd.to_datetime(df_FB_['Date'])
	df_FB_= df_FB_.sort_values('Date')
	# plot data source  
	plot_data(df_FB_)
	# preprocess data 
	train_data,test_data, scaler= preprocess_data(df_FB_)
	df_FB_.head(3)
	# train the model & plot result
	model_avg_prev_scalar(train_data,test_data)






