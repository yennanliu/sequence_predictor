
# python 3 

# https://keras.io/callbacks/
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

from keras.callbacks import Callback


#print(callbacks.history.history.keys())


def show_model_fitting_history(model,X,Y):
		# Fit the model
	history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
	# list all data in history
	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	return history








