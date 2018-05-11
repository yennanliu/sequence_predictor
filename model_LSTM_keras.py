# python 3 

# ops 
import pandas as pd 
import numpy as np
import math
#import matplotlib.pyplot as plt
#%pylab inline


# DL
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU



#---------------------------------------


#########  data preprocess #########


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



def get_train_data(df):
	#df_expo = df[df.drop_off_addr == 'expo']
	#dataset = df_expo.groupby('timestamp_live_vec_table')\
	#				 .median()[['lag_idle_day']]\
	#				 .values
	#dataset = df[['Volume','High','Open']].values.astype('float32')
	dataset = df[['Open']].values.astype('float32')
	#print (shape(dataset))
	print (dataset) 
	return dataset



# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


#---------------------------------------

#########  V1 ARCHITECTURE #########

# V1 model 
def one_input_LSTM_model_1(dataset):
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	# reshape into X=t and Y=t+1
	look_back = 1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	# create and fit the LSTM network
	model = Sequential()
	# expected input data shape: (batch_size, input_shape=(timesteps, data_dim))
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	# shift train predictions for plotting
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	print (model.summary())
	return  dataset,trainPredict,testPredict,trainPredictPlot,testPredictPlot






def one_input_LSTM_model_2(dataset):
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	# reshape into X=t and Y=t+1
	look_back = 1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	# create and fit the LSTM network
	model = Sequential()
	"""
	https://github.com/keras-team/keras/issues/160
	A LSTM layer, as per the docs, will return the last vector by default rather than the entire sequence. 
	In order to return the entire sequence (which is necessary to be able to stack LSTM), 
	use the constructor argument return_sequences=True.
	"""
	# expected input data shape: (batch_size, input_shape=(timesteps, data_dim))
	model.add(LSTM(4, return_sequences=True,input_shape=(1, look_back)))
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	# shift train predictions for plotting
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	print (model.summary())
	return  dataset,trainPredict,testPredict,trainPredictPlot,testPredictPlot


def one_input_LSTM_model_3(dataset):
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	# reshape into X=t and Y=t+1
	look_back = 1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	# create and fit the LSTM network
	model = Sequential()
	"""
	https://github.com/keras-team/keras/issues/3522
	A LSTM layer, as per the docs, will return the last vector by default rather than the entire sequence. 
	In order to return the entire sequence (which is necessary to be able to stack LSTM), 
	use the constructor argument return_sequences=True.
	"""
	# expected input data shape: (batch_size, input_shape=(timesteps, data_dim))
	model.add(LSTM(4, return_sequences=True,input_shape=(1, look_back)))
	# add 5 stack hidden layers 
	# -------------------------
	model.add(LSTM(4, return_sequences=True,input_shape=(1, look_back)))
	model.add(LSTM(4, return_sequences=True,input_shape=(1, look_back)))
	model.add(LSTM(4, return_sequences=True,input_shape=(1, look_back)))
	model.add(LSTM(4, return_sequences=True,input_shape=(1, look_back)))
	model.add(LSTM(4, return_sequences=True,input_shape=(1, look_back)))
	# -------------------------
	model.add(LSTM(4))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	# shift train predictions for plotting
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	print (model.summary())
	return  dataset,trainPredict,testPredict,trainPredictPlot,testPredictPlot


#---------------------------------------

#########  V2 ARCHITECTURE #########

# V2 model 
# credit  
# https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-1-2-correct-time-series-forecasting-backtesting-9776bfd9e589


def dev_model_1(dataset):
	print (dataset)
	model = Sequential()
	model.add(Dense(dataset.shape[0], input_shape=(1, 1))) 
	model.add(BatchNormalization()) 
	model.add(LeakyReLU()) 
	model.add(Flatten())
	model.add(Dense(1)) 
	model.add(Activation('softmax'))
	return model 


def dev_model_2():
    model = Sequential() 
    model.add(Dense(64, input_dim=30)) 
    model.add(BatchNormalization()) 
    model.add(LeakyReLU()) 
    model.add(Dense(16)) 
    model.add(BatchNormalization()) 
    model.add(LeakyReLU()) 
    model.add(Dense(2)) 
    model.add(Activation('softmax'))
    return model 

def dev_model_3():
    model = Sequential() 
    model.add(Dense(64, input_dim=30,activity_regularizer=regularizers.l2(0.01))) 
    model.add(BatchNormalization()) 
    model.add(LeakyReLU()) 
    model.add(Dense(16,activity_regularizer=regularizers.l2(0.01))) 
    model.add(BatchNormalization()) 
    model.add(LeakyReLU()) 
    model.add(Dense(2)) 
    model.add(Activation('softmax'))
    return model 

def dev_model_4():
    model = Sequential()  
    model.add(Dense(64, input_dim=30,activity_regularizer=regularizers.l2(0.01))) 
    model.add(BatchNormalization()) 
    model.add(LeakyReLU()) 
    model.add(Dropout(0.5)) 
    model.add(Dense(16,activity_regularizer=regularizers.l2(0.01))) 
    model.add(BatchNormalization()) 
    model.add(LeakyReLU()) 
    model.add(Dense(2)) 
    model.add(Activation('softmax'))
    return model 





# V2 ARCHITECTURE runner 

def V2_model_runner(dataset, model):
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1])) 
    ####### import pre-defined model #######
    print (model) 
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    print (model.summary())
    return  dataset,trainPredict,testPredict,trainPredictPlot,testPredictPlot



#---------------------------------------


if __name__ == '__main__':
	df_FB = get_data('FB')
	df_FB_ = col_fix(df_FB)
	dataset = get_train_data(df_FB_)

	######## RUN V1 ARCHITECTURE  ######## 
	
	#dataset,trainPredict,testPredict,trainPredictPlot,testPredictPlot = one_input_LSTM_model_1(dataset)
	#dataset,trainPredict,testPredict,trainPredictPlot,testPredictPlot = one_input_LSTM_model_3(dataset)
	
	######## RUN V2 ARCHITECTURE ######## 
	model=dev_model_1(dataset)
	dataset,trainPredict,testPredict,trainPredictPlot,testPredictPlot = V2_model_runner(dataset,model )


	# plot 
	#plt.plot(trainPredictPlot)
	#plt.plot(testPredictPlot)
	#plt.plot( df_FB_[['Open']])





