#python 3 


# V2 OP
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, RNN,GRU,SimpleRNNCell
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
import keras.backend as K
import keras
import emoji
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# UDF 
from emo_utils import *



#-------------------------------------------------
# config 
np.random.seed(1)


#-------------------------------------------------
# help func 

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j += 1
            
    ### END CODE HERE ###
    
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer




class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]
 
#-------------------------------------------------
# model  
def Emojify_RNN_model(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(input_shape, dtype = 'int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    # https://stackoverflow.com/questions/45989610/setting-up-the-input-on-an-rnn-in-keras
    # Desired result is a sequence with same length, we will use return_sequences=True. (Else, you'd get only one result).
    # keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
    cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
    X = RNN(cells, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = Dense(5)(X)
    X = Activation('sigmoid')(X)
    model = Model(inputs=sentence_indices, outputs=X)  
    return model





#-------------------------------------------------

# model run func 

def main():
    # load the data 
    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/tesss.csv')
    maxLen = len(max(X_train, key=len).split())
    Y_oh_train = convert_to_one_hot(Y_train, C = 5)
    Y_oh_test = convert_to_one_hot(Y_test, C = 5)
    """
    # need to download the extra dataset via 
    # https://www.kaggle.com/watts2/glove6b50dtxt/version/1#
    # https://nlp.stanford.edu/projects/glove/
    # via kaggle API : kaggle datasets download -d watts2/glove6b50dtxt
    """
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
    #### load RNN model
    model = Emojify_RNN_model((maxLen,), word_to_vec_map, word_to_index)
    ####
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C = 5)
    history = model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)
    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C = 5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    print ('####  loss    (train data) ####')
    print (history.history['loss'])
    print ('####  accuracy   (train data)  ####')
    print (history.history['acc'])
    print()
    print ('####  Test accuracy    (test data) ####')
    print("Test accuracy = ", acc)
    # This code allows you to see the mislabelled examples
    C = 5
    y_test_oh = np.eye(C)[Y_test.reshape(-1)]
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
    pred = model.predict(X_test_indices)
    # collect X_test pred 
    X_test_pred = []
    print ('####  pred output  (test data) ####')
    for i in range(len(X_test)):
        x = X_test_indices
        num = np.argmax(pred[i])
        X_test_pred.append(num)
    #if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())
    print ('####  confusion_matrix (test data) ####')
    print (confusion_matrix(np.array(X_test_pred),Y_test))
    #plot_confusion_matrix(np.array(X_test_pred),Y_test)




#-------------------------------------------------



if __name__ == '__main__':
	main()






