# python 3
# modify from 
#https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb



# OP
import pandas as pd 
import numpy as np 
import os
from sklearn.preprocessing import MinMaxScaler
# DL 
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


# -----------------------------------
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

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)



def loss_mse_warmup(y_true, y_pred):
	"""
	Calculate the Mean Squared Error between y_true and y_pred,
	but ignore the beginning "warmup" part of the sequences.

	y_true is the desired output.
	y_pred is the model's output.
	"""

	# The shape of both input tensors are:
	# [batch_size, sequence_length, num_y_signals].

	# Ignore the "warmup" parts of the sequences
	# by taking slices of the tensors.
	y_true_slice = y_true[:, warmup_steps:, :]
	y_pred_slice = y_pred[:, warmup_steps:, :]

	# These sliced tensors both have this shape:
	# [batch_size, sequence_length - warmup_steps, num_y_signals]

	# Calculate the MSE loss for each value in these tensors.
	# This outputs a 3-rank tensor of the same shape.
	loss = tf.losses.mean_squared_error(labels=y_true_slice,
	                                    predictions=y_pred_slice)

	# Keras may reduce this across the first axis (the batch)
	# but the semantics are unclear, so to be sure we use
	# the loss across the entire tensor, we reduce it to a
	# single scalar with the mean function.
	loss_mean = tf.reduce_mean(loss)

	return loss_mean


# -----------------------------------


if __name__ == '__main__':
	col_name ='Open'
	df_FB = get_data('FB')
	df_FB_ = col_fix(df_FB)
	dataset = df_FB_[['Open']].values.astype('float32')
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
	validation_data = (np.expand_dims(testX, axis=0),
                   	np.expand_dims(testY, axis=0))
	print('trainX.shape :',trainX.shape)
	print('trainY.shape :' , trainY.shape) 
	# generate batch 
	batch_size = 256
	sequence_length  = 10
	num_x_signals = trainX.shape[1] 
	num_y_signals = trainY.shape[0]
	train_split = 0.8
	num_train = int(train_split * len(trainX))

	#generator = batch_generator(batch_size=batch_size,
    #                        sequence_length=sequence_length)

	#x_batch, y_batch = next(generator)
	#print(x_batch.shape)
	#print(y_batch.shape)


	# ---------- DATA PREPROCESS ----------


	# ---------- tf RNN model  ----------
	model = Sequential()
	model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))
	model.add(Dense(num_y_signals, activation='sigmoid'))
	warmup_steps = 50
	optimizer = RMSprop(lr=1e-3)
	model.compile(loss=loss_mse_warmup, optimizer=optimizer)
	print (model.summary())
	model.fit_generator(generator=(trainX,trainY),
                    epochs=20,
                    steps_per_epoch=100)







