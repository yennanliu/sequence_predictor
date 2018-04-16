# python 3 


# ops 
import pandas as pd 
import numpy as np 
# DL 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.callbacks import Callback



# help function 
#---------------------------------------

# data prepare 
def get_data(fine_name):
	print (fine_name)
	df = pd.read_csv('data/{}.csv'.format(fine_name))
	print (df.head())
	return df  

def col_fix(df):
    df = df.drop('Unnamed: 0', axis=1) 
    print (df.columns)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    print (df.head())
    return df 




# model DL 

def RNN_model_1(df,col_name):
	# fitting with scaled data 
	model = Sequential()
	model.add(Dense(500, input_shape = (1, )))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(250))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('softmax'))
	model.compile(optimizer='adam', loss='mse')
# --------------
	### only get col_name as input / target data values 
	df_ = df.copy()
	x__ = df_[col_name]
	Y_train = df_[col_name]
	params = []
	for xt in x__: # taking training data from unscaled array
		xt = np.array(xt)
		#print (xt)
		mean_ = xt.mean()
		scale_ = x__.std()
		params.append([mean_, scale_])

	predicted = model.predict(x__)
	new_predicted = []

	# restoring data
	for pred, par in zip(predicted, params):
		a = pred*par[1]
		a += par[0]
		new_predicted.append(a)

# --------------
	new_predicted_ = np.array(new_predicted).flatten()
	new_predicted_ = pd.DataFrame(new_predicted_, dtype='str')
	output=pd.DataFrame()
	output['actual'] =Y_train
	output['predict'] =new_predicted_
	output['predict']=output['predict'].astype(float)
	# print model architecture
	print (model.summary())
	# print prediction  
	print (output)
	return output 



def RNN_model_2(df,col_name):
	# fitting with scaled data 
	model = Sequential()
	model.add(Dense(500, input_shape = (1, )))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(500))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('relu'))
	model.add(Dense(500, input_shape = (1, )))
	model.add(Activation('softmax'))
	model.add(Dropout(0.25))
	model.compile(optimizer='adam', loss='mse')
# --------------
	### only get col_name as input / target data values 
	df_ = df.copy()
	x__ = df_[col_name]
	Y_train = df_[col_name]
	params = []
	for xt in x__: # taking training data from unscaled array
		xt = np.array(xt)
		#print (xt)
		mean_ = xt.mean()
		scale_ = x__.std()
		params.append([mean_, scale_])

	predicted = model.predict(x__)
	new_predicted = []

	# restoring data
	for pred, par in zip(predicted, params):
		a = pred*par[1]
		a += par[0]
		new_predicted.append(a)

# --------------
	new_predicted_ = np.array(new_predicted).flatten()
	new_predicted_ = pd.DataFrame(new_predicted_, dtype='str')
	output=pd.DataFrame()
	output['actual'] =Y_train
	output['predict'] =new_predicted_
	output['predict']=output['predict'].astype(float)
	# print model architecture
	print (model.summary())
	# print prediction  
	print (output)
	return output 


#---------------------------------------



if __name__ == '__main__':
	df_FB = get_data('FB')
	df_FB_ = col_fix(df_FB)
	#RNN_model_1(df_FB_,'Open')
	RNN_model_2(df_FB_,'Open')

	







