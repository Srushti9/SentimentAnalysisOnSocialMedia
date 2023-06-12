# -*- coding: utf-8 -*-

pip install tensorflow_text

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from google.colab import drive
drive.mount('/content/drive')

pwd

import os
os.chdir('./drive/My Drive')

import pandas as pd
all = pd.read_csv('/content/drive/My Drive/sst5_dataset.csv')
all = all[['text','target']]
all['text'] = all['text'].str.replace('_label_', '')
all['target'] = all['target'].astype(int).astype('category')
all.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(all['text'],all['target'],stratify=all['target'])

y_train = tf.one_hot(y_train,5)
y_test = tf.one_hot(y_test,5)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#x_train=x_train[0:200]
#x_test=x_test[0:50,]
#y_train=y_train[0:200]
#y_test=y_test[0:50,]

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def get_sentence_embedding(sentence):
  preprocessed_text = bert_preprocess(sentence)
  #print(preprocessed_text)
  return bert_encoder(preprocessed_text)['pooled_output']

get_sentence_embedding(["happy"])

# Bert layers
text_input=tf.keras.layers.Input(shape=(),dtype=tf.string,name="text")
preprocessed_text=bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)
# Neural Network
l = tf.keras.layers.Dense(300,activation="tanh",name='reduction',dtype='float32')(outputs['pooled_output'])
l = tf.keras.layers.Dense(150,activation="tanh",name="dense",dtype='float32')(l)
l = tf.keras.layers.Dense(5,activation="softmax",name="output",dtype='float32')(l)

#construct final model
model=tf.keras.Model(inputs=[text_input],outputs=l)

model.summary()

#METRICS = [
 #           tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
 #           tf.keras.metrics.Precision(name='precision'),
 #           tf.keras.metrics.Recall(name='recall')
 #];

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['Accuracy'])

history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=8)

history.history.keys()

history.history['Accuracy']

history.history['val_Accuracy']

model.save("./sst_BERT_model_acc_31_val_28.h5")

y_hat = model.predict((x_test))

from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
cm=confusion_matrix(list(map(lambda x: np.argmax(x), y_test)), 
                    list(map(lambda x: np.argmax(x), y_hat))) 
sns.set(rc = {'figure.figsize':(12,6)})
sns.heatmap(cm,annot=True,cmap='Blues',fmt='g')

from sklearn.metrics import classification_report
cr = classification_report(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), y_hat)))
print(cr)

"""CNN+BERT"""

# Converting text to sequence of numbers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
tokenizer = Tokenizer()
# Assign a number to word
tokenizer.fit_on_texts(all['text'])
# Convert text to sequence of number 
sequences = tokenizer.texts_to_sequences(all['text'])

# Identifying max length of reviews
 total_length = 0
 avg_length = 0
 count = 0;
 max_length = 0
 for text in sequences:
   total_length = total_length + len(text)
   count = count + 1
   length = len(text)
   if length > max_length :
     max_length = length
 print(max_length)
 avg_length = total_length / count
 print(avg_length)

# Pad reviews to make reviews of equal size
max_length = 7
from tensorflow.keras.preprocessing import sequence
sequences = sequence.pad_sequences(sequences, maxlen=max_length,padding='post')
sequences.shape

# use 200 samples for training and all for testing
x_train=x_train[0:200]
x_train_global = sequences[:200,:]
x_test_global = sequences[8891:]
print(x_train.shape)
import numpy
# a = numpy.array(x_train_global)
# b = numpy.array(y_test_global)
# print(a.shape)
# print(b.shape)

# Create training data and testing data
x_train_global = sequences[:200,:]
x_test_global = sequences[0:50,:]

print(x_train.shape)
print(x_test.shape)
import numpy
a = numpy.array(y_train)
b = numpy.array(y_test)
print(a.shape)
print(b.shape)

# print word index dictionary
word_index = tokenizer.word_index
print(tokenizer.word_index)
# print the number of unique tokens
print('Found %s unique tokens.' % len(word_index))

import gensim
path1='/content/drive/MyDrive/GoogleNews-vectors-negative300.bin'
path2='/content/drive/MyDrive/refine_mywordembeddings_Intensity.bin'
mywordembeddings = gensim.models.KeyedVectors.load_word2vec_format(path2, binary=True)

unique_words = len(word_index)
total_words = unique_words + 1
skipped_words = 0
embedding_dim = 300  
embedding_matrix = np.zeros((total_words, embedding_dim))
for word, index in tokenizer.word_index.items():
  try:
    embedding_vector = mywordembeddings[word]
  except:
    embedding_vector = mywordembeddings['the']
    skipped_words = skipped_words+1
    pass
  if embedding_vector is not None:
    embedding_matrix[index] = embedding_vector
print(skipped_words)
print(embedding_matrix.shape)

from keras.layers import *
# create the embedding layer
embedding_layer = Embedding(total_words, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)#(i.e., the number of words in each sentence(input length))

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers as initializers, regularizers, constraints
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)


        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPool2D, MaxPool1D, Flatten, Dense, Dropout

from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
#K.set_image_dim_ordering('th')

input = Input(shape=(max_length,),dtype='float32')

# create the embedding layer
embedding_layer = Embedding(total_words, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)(input)


conv_0 = Conv1D(8, kernel_size=(3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(embedding_layer)
conv_1 = Conv1D(8, kernel_size=(4), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(embedding_layer)
conv_2 = Conv1D(8, kernel_size=(5), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(embedding_layer)

maxpool_0 = MaxPooling1D(pool_size=(4),padding='same')(conv_0)
maxpool_1 = MaxPool1D(pool_size=(4),  padding='same')(conv_1)
maxpool_2 = MaxPool1D(pool_size=(4), padding='same')(conv_2)

concatenated_tensor_2 = Concatenate(axis=-1)([maxpool_0,maxpool_1,maxpool_2])
flatten = GlobalAveragePooling1D()(concatenated_tensor_2)
output = Dense(units=150, activation='relu')(flatten)
testoutput = Dense(units=1, activation='sigmoid')(output)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

import numpy as np
y_train = np.argmax(y_train, axis=1)

model=tf.keras.Model(inputs=input,outputs=testoutput)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_global,y_train,batch_size=2,epochs=2)

# Bert layers
text_input=tf.keras.layers.Input(shape=(),dtype=tf.string,name="text")
preprocessed_text=bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)
# Neural Network
l = tf.keras.layers.Dense(300,activation="tanh", name='reduction')(outputs['pooled_output'])
l = tf.keras.layers.Dense(150,activation="tanh",name="dense2")(l)
concatenated_tensor = Concatenate(axis=1)([l,output])
d1 = Dense(units=150, activation='relu')(concatenated_tensor)
final_output = Dense(units=5, activation='softmax')(d1)

#construct final model

model=tf.keras.Model(inputs=[text_input,input],outputs=final_output)
model.summary()

from keras.utils import to_categorical

y_train_cat = to_categorical(y_train)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history1=model.fit((x_train,x_train_global),y_train_cat,validation_split=0.1,epochs=8);

y_hat = model.predict((x_test,x_test_global))

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), y_hat)))
sns.set(rc = {'figure.figsize':(12,6)})
sns.heatmap(cm,annot=True,cmap='Blues',fmt='g')

from sklearn.metrics import classification_report
cr = classification_report(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), y_hat)))
print(cr)

history1.history.keys()

history1.history['accuracy']

history1.history['val_accuracy']

model.save('./sst5_BERTCNN_P19_R25_F120')

import matplotlib.pyplot as plt
BERT_Accuracy =  [0.3199999928474426, 0.30000001192092896, 0.25999999046325684, 0.2800000011920929, 0.30000001192092896, 0.25999999046325684, 0.30000001192092896, 0.2800000011920929]
BERT_CNN_Accuracy = [0.18333333730697632, 0.25, 0.2666666805744171, 0.25555557012557983, 0.28333333134651184, 0.3611111044883728, 0.33888888359069824, 0.3888888955116272]
plt.plot(BERT_Accuracy)
plt.plot(BERT_CNN_Accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['BERT','BERT+CNN'],loc='upper left')

import matplotlib.pyplot as plt
BERT_Accuracy = [ 0.3199999928474426, 0.30000001192092896, 0.25999999046325684, 0.2800000011920929, 0.30000001192092896, 0.25999999046325684, 0.30000001192092896, 0.2800000011920929 ]
BERT_CNN_Accuracy =  [0.4000000059604645, 0.10000000149011612, 0.4000000059604645, 0.3499999940395355, 0.10000000149011612, 0.4000000059604645, 0.20000000298023224, 0.4000000059604645]
plt.plot(BERT_Accuracy)
plt.plot(BERT_CNN_Accuracy)
plt.legend(['BERT_Loss','BERT_CNN_Loss'],loc='upper right')

import matplotlib.pyplot as plt
BERT_Accuracy =  [0.375,
 0.5299999713897705,
 0.4950000047683716,
 0.5350000262260437,
 0.5199999809265137,
 0.5400000214576721,
 0.5350000262260437,
 0.5400000214576721]
BERT_CNN_Accuracy = [0.4444444477558136,
 0.4166666567325592,
 0.4555555582046509,
 0.550000011920929,
 0.5388888716697693,
 0.5611110925674438,
 0.5722222328186035,
 0.5277777910232544]
plt.plot(BERT_Accuracy)
plt.plot(BERT_CNN_Accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['BERT','BERT+CNN'],loc='upper left')

import matplotlib.pyplot as plt
BERT_Accuracy = [0.47999998927116394,
 0.5400000214576721,
 0.5,
 0.5199999809265137,
 0.5199999809265137,
 0.5199999809265137,
 0.5400000214576721,
 0.5199999809265137]
BERT_CNN_Accuracy =  [0.3499999940395355,
 0.550000011920929,
 0.5,
 0.25,
 0.5,
 0.3499999940395355,
 0.4000000059604645,
 0.6000000238418579]
plt.plot(BERT_Accuracy)
plt.plot(BERT_CNN_Accuracy)
plt.legend(['BERT_Loss','BERT_CNN_Loss'],loc='upper right')

