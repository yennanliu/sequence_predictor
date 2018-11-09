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
	######### DATA PREPROCESS V2  #########
	x_data = df_FB_.iloc[:,1:].values
	y_data = df_FB_[['Open']].values.astype('float32')
	# split into train and test sets
	num_data = len(x_data)
	train_split = 0.9
	num_train = int(train_split * num_data)
	num_test = num_data - num_train
	x_train = x_data[0:num_train]
	x_test = x_data[num_train:]
	y_train = y_data[0:num_train]
	y_test = y_data[num_train:]
	num_x_signals = x_data.shape[1]
	num_y_signals = y_data.shape[1]
	# RESCALE DATA 
	x_scaler = MinMaxScaler()
	x_train_scaled = x_scaler.fit_transform(x_train)
	x_test_scaled = x_scaler.transform(x_test)
	y_scaler = MinMaxScaler()
	y_train_scaled = y_scaler.fit_transform(y_train)
	y_test_scaled = y_scaler.transform(y_test)
	# RECHECK DATA SHAPE 
	print(x_train_scaled.shape)
	print(y_train_scaled.shape)
	# GET BATCH DATA 
	batch_size = 4
	sequence_length = 24 
	sequence_length
	generator = batch_generator(batch_size=batch_size,
	                            sequence_length=sequence_length)
	x_batch, y_batch = next(generator)
	print(x_batch.shape)
	print(y_batch.shape)
	batch = 0   # First sequence in the batch.
	signal = 0  # First signal from the 20 input-signals.
	seq = x_batch[batch, :, signal]
	# VALIDATION SET 
	validation_data = (np.expand_dims(x_test_scaled, axis=0),
	                   np.expand_dims(y_test_scaled, axis=0))

	### --------------  MODEL  --------------###
	model = Sequential()
	model.add(GRU(units=512,
	              return_sequences=True,
	              input_shape=(None, num_x_signals,)))

	model.add(Dense(num_y_signals, activation='sigmoid'))

	### -------------- MODEL TRAIN (TRAIN SET)  --------------### 
	warmup_steps = 30
	optimizer = RMSprop(lr=1e-3)
	model.compile(loss=loss_mse_warmup, optimizer=optimizer)
	model.summary()
	callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
	                                       factor=0.1,
	                                       min_lr=1e-4,
	                                       patience=0,
	                                       verbose=1)
	callbacks = [callback_reduce_lr]
	model.fit_generator(generator=generator,
	                    epochs=5,
	                    steps_per_epoch=10,
	                    validation_data=validation_data,
	                    callbacks=callbacks)


	### -------------- MODEL TEST (TEST SET)  -------------- ###

	result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
	                        y=np.expand_dims(y_test_scaled, axis=0))
	print("loss (test-set):", result)








